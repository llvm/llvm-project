// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir -mno-constructor-aliases %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -mno-constructor-aliases -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -mno-constructor-aliases -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void some_function() noexcept;

struct out_of_line_destructor {
    int prevent_tail_padding_reuse;
    ~out_of_line_destructor();
};

out_of_line_destructor::~out_of_line_destructor() {
    some_function();
}

// CIR: !rec_out_of_line_destructor = !cir.record<struct "out_of_line_destructor" {!s32i}>

// CIR: cir.func dso_local @_ZN22out_of_line_destructorD2Ev(%{{.+}}: !cir.ptr<!rec_out_of_line_destructor>
// CIR:   cir.call @_Z13some_functionv() nothrow : () -> () 
// CIR:   cir.return 

// LLVM: define dso_local void @_ZN22out_of_line_destructorD2Ev(ptr %{{.+}})
// LLVM:   call void @_Z13some_functionv()
// LLVM:   ret void

// OGCG: define dso_local void @_ZN22out_of_line_destructorD2Ev(ptr {{.*}}%{{.+}})
// OGCG:   call void @_Z13some_functionv()
// OGCG:   ret void

// CIR: cir.func dso_local @_ZN22out_of_line_destructorD1Ev(%{{.+}}: !cir.ptr<!rec_out_of_line_destructor>
// CIR:  cir.call @_ZN22out_of_line_destructorD2Ev(%{{.*}}) nothrow : (!cir.ptr<!rec_out_of_line_destructor>)
// CIR:  cir.return

// LLVM: define dso_local void @_ZN22out_of_line_destructorD1Ev(ptr %{{.+}})
// LLVM:   call void @_ZN22out_of_line_destructorD2Ev
// LLVM:   ret void

// OGCG: define dso_local void @_ZN22out_of_line_destructorD1Ev(ptr {{.*}}%{{.+}})
// OGCG:   call void @_ZN22out_of_line_destructorD2Ev
// OGCG:   ret void

struct inline_destructor {
    int prevent_tail_padding_reuse;
    ~inline_destructor() noexcept(false) {
        some_function();
    }
};

// This inline destructor is not odr-used in this TU.
// Make sure we don't emit a definition

// CIR-NOT: cir.func {{.*}}inline_destructor{{.*}}
// LLVM-NOT: define {{.*}}inline_destructor{{.*}}
// OGCG-NOT: define {{.*}}inline_destructor{{.*}}

struct array_element {~array_element();};
void test_array_destructor() {
  array_element arr[5]{};
}

// CIR: cir.func dso_local @_Z21test_array_destructorv()
// CIR:   %[[ARR:.*]] = cir.alloca !cir.array<!rec_array_element x 5>, !cir.ptr<!cir.array<!rec_array_element x 5>>, ["arr", init]
// CIR:   %[[ARR_PTR:.*]] = cir.alloca !cir.ptr<!rec_array_element>, !cir.ptr<!cir.ptr<!rec_array_element>>, ["arrayinit.temp", init]
// CIR:   %[[BEGIN:.*]] = cir.cast array_to_ptrdecay %[[ARR]] : !cir.ptr<!cir.array<!rec_array_element x 5>>
// CIR:   cir.store{{.*}} %[[BEGIN]], %[[ARR_PTR]]
// CIR:   %[[FIVE:.*]] = cir.const #cir.int<5> : !s64i
// CIR:   %[[ARR_END:.*]] = cir.ptr_stride %[[BEGIN]], %[[FIVE]] : (!cir.ptr<!rec_array_element>, !s64i)
// CIR:   cir.do {
// CIR:     %[[ARR_CUR:.*]] = cir.load{{.*}} %[[ARR_PTR]]
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR:     %[[ARR_NEXT:.*]] = cir.ptr_stride %[[ARR_CUR]], %[[ONE]] : (!cir.ptr<!rec_array_element>, !s64i)
// CIR:     cir.store{{.*}} %[[ARR_NEXT]], %[[ARR_PTR]] : !cir.ptr<!rec_array_element>, !cir.ptr<!cir.ptr<!rec_array_element>>
// CIR:     cir.yield
// CIR:   } while {
// CIR:     %[[ARR_CUR:.*]] = cir.load{{.*}} %[[ARR_PTR]]
// CIR:     %[[CMP:.*]] = cir.cmp(ne, %[[ARR_CUR]], %[[ARR_END]])
// CIR:     cir.condition(%[[CMP]])
// CIR:   }
// CIR:   %[[FOUR:.*]] = cir.const #cir.int<4> : !u64i
// CIR:   %[[BEGIN:.*]] = cir.cast array_to_ptrdecay %[[ARR]] : !cir.ptr<!cir.array<!rec_array_element x 5>>
// CIR:   %[[END:.*]] = cir.ptr_stride %[[BEGIN]], %[[FOUR]] : (!cir.ptr<!rec_array_element>, !u64i)
// CIR:   %[[ARR_PTR:.*]] = cir.alloca !cir.ptr<!rec_array_element>, !cir.ptr<!cir.ptr<!rec_array_element>>, ["__array_idx"]
// CIR:   cir.store %[[END]], %[[ARR_PTR]]
// CIR:   cir.do {
// CIR:     %[[ARR_CUR:.*]] = cir.load{{.*}} %[[ARR_PTR]]
// CIR:     cir.call @_ZN13array_elementD1Ev(%[[ARR_CUR]]) nothrow : (!cir.ptr<!rec_array_element>) -> ()
// CIR:     %[[NEG_ONE:.*]] = cir.const #cir.int<-1> : !s64i
// CIR:     %[[ARR_NEXT:.*]] = cir.ptr_stride %[[ARR_CUR]], %[[NEG_ONE]] : (!cir.ptr<!rec_array_element>, !s64i)
// CIR:     cir.store %[[ARR_NEXT]], %[[ARR_PTR]]
// CIR:     cir.yield
// CIR:   } while {
// CIR:     %[[ARR_CUR:.*]] = cir.load{{.*}} %[[ARR_PTR]]
// CIR:     %[[CMP:.*]] = cir.cmp(ne, %[[ARR_CUR]], %[[BEGIN]])
// CIR:     cir.condition(%[[CMP]])
// CIR:   }

// LLVM: define{{.*}} void @_Z21test_array_destructorv()
// LLVM:   %[[ARR:.*]] = alloca [5 x %struct.array_element]
// LLVM:   %[[TMP:.*]] = alloca ptr
// LLVM:   %[[ARR_PTR:.*]] = getelementptr %struct.array_element, ptr %[[ARR]], i32 0
// LLVM:   store ptr %[[ARR_PTR]], ptr %[[TMP]]
// LLVM:   %[[END_PTR:.*]] = getelementptr %struct.array_element, ptr %[[ARR_PTR]], i64 5
// LLVM:   br label %[[INIT_LOOP_BODY:.*]]
// LLVM: [[INIT_LOOP_NEXT:.*]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[TMP]]
// LLVM:   %[[CMP:.*]] = icmp ne ptr %[[CUR]], %[[END_PTR]]
// LLVM:   br i1 %[[CMP]], label %[[INIT_LOOP_BODY]], label %[[INIT_LOOP_END:.*]]
// LLVM: [[INIT_LOOP_BODY]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[TMP]]
// LLVM:   %[[NEXT:.*]] = getelementptr %struct.array_element, ptr %[[CUR]], i64 1
// LLVM:   store ptr %[[NEXT]], ptr %[[TMP]]
// LLVM:   br label %[[INIT_LOOP_NEXT:.*]]
// LLVM: [[INIT_LOOP_END]]:
// LLVM:   %[[ARR_BEGIN:.*]] = getelementptr %struct.array_element, ptr %[[ARR]], i32 0
// LLVM:   %[[ARR_END:.*]] = getelementptr %struct.array_element, ptr %[[ARR_BEGIN]], i64 4
// LLVM:   %[[ARR_CUR:.*]] = alloca ptr
// LLVM:   store ptr %[[ARR_END]], ptr %[[ARR_CUR]]
// LLVM:   br label %[[DESTROY_LOOP_BODY:.*]]
// LLVM: [[DESTROY_LOOP_NEXT:.*]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[ARR_CUR]]
// LLVM:   %[[CMP:.*]] = icmp ne ptr %[[CUR]], %[[ARR_BEGIN]]
// LLVM:   br i1 %[[CMP]], label %[[DESTROY_LOOP_BODY]], label %[[DESTROY_LOOP_END:.*]]
// LLVM: [[DESTROY_LOOP_BODY]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[ARR_CUR]]
// LLVM:   call void @_ZN13array_elementD1Ev(ptr %[[CUR]])
// LLVM:   %[[PREV:.*]] = getelementptr %struct.array_element, ptr %[[CUR]], i64 -1
// LLVM:   store ptr %[[PREV]], ptr %[[ARR_CUR]]
// LLVM:   br label %[[DESTROY_LOOP_NEXT]]
// LLVM: [[DESTROY_LOOP_END]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z21test_array_destructorv()
// OGCG: entry:
// OGCG:   %[[ARR:.*]] = alloca [5 x %struct.array_element]
// OGCG:   %[[ARRAYINIT_END:.*]] = getelementptr inbounds %struct.array_element, ptr %[[ARR]], i64 5
// OGCG:   br label %[[INIT_LOOP_BODY:.*]]
// OGCG: [[INIT_LOOP_BODY]]:
// OGCG:   %[[CUR:.*]] = phi ptr [ %[[ARR]], %entry ], [ %[[NEXT:.*]], %[[INIT_LOOP_BODY]] ]
// OGCG:   %[[NEXT]] = getelementptr inbounds %struct.array_element, ptr %[[CUR]], i64 1
// OGCG:   %[[CMP:.*]] = icmp eq ptr %[[NEXT]], %[[ARRAYINIT_END]]
// OGCG:   br i1 %[[CMP]], label %[[INIT_LOOP_END:.*]], label %[[INIT_LOOP_BODY]]
// OGCG: [[INIT_LOOP_END:.*]]:
// OGCG:   %[[BEGIN:.*]] = getelementptr inbounds [5 x %struct.array_element], ptr %[[ARR]], i32 0, i32 0
// OGCG:   %[[END:.*]] = getelementptr inbounds %struct.array_element, ptr %[[BEGIN]], i64 5
// OGCG:   br label %[[DESTROY_LOOP_BODY:.*]]
// OGCG: [[DESTROY_LOOP_BODY:.*]]:
// OGCG:   %[[CUR:.*]] = phi ptr [ %[[END]], %[[INIT_LOOP_END]] ], [ %[[PREV:.*]], %[[DESTROY_LOOP_BODY]] ]
// OGCG:   %[[PREV]] = getelementptr inbounds %struct.array_element, ptr %[[CUR]], i64 -1
// OGCG:   call void @_ZN13array_elementD1Ev(ptr {{.*}} %[[PREV]])
// OGCG:   %[[CMP:.*]] = icmp eq ptr %[[PREV]], %[[BEGIN]]
// OGCG:   br i1 %[[CMP]], label %[[DESTROY_LOOP_END:.*]], label %[[DESTROY_LOOP_BODY]]
// OGCG: [[DESTROY_LOOP_END:.*]]:
// OGCG:   ret void
