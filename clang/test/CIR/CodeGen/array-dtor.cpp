// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefixes=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct S {
    ~S();
};

void test_cleanup_array() {
    S s[42];
}

// CIR: cir.func{{.*}} @_Z18test_cleanup_arrayv()
// CIR:   %[[S:.*]] = cir.alloca !cir.array<!rec_S x 42>, !cir.ptr<!cir.array<!rec_S x 42>>, ["s"]
// CIR:   cir.array.dtor %[[S]] : !cir.ptr<!cir.array<!rec_S x 42>> {
// CIR:   ^bb0(%arg0: !cir.ptr<!rec_S>
// CIR:     cir.call @_ZN1SD1Ev(%arg0) nothrow : (!cir.ptr<!rec_S>) -> ()
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.return

// LLVM: define{{.*}} void @_Z18test_cleanup_arrayv()
// LLVM:   %[[ARRAY:.*]] = alloca [42 x %struct.S]
// LLVM:   %[[START:.*]] = getelementptr %struct.S, ptr %[[ARRAY]], i32 0
// LLVM:   %[[END:.*]] = getelementptr %struct.S, ptr %[[START]], i64 41
// LLVM:   %[[ITER:.*]] = alloca ptr
// LLVM:   store ptr %[[END]], ptr %[[ITER]]
// LLVM:   br label %[[LOOP:.*]]
// LLVM: [[COND:.*]]:
// LLVM:   %[[CURRENT_CHECK:.*]] = load ptr, ptr %[[ITER]]
// LLVM:   %[[DONE:.*]] = icmp ne ptr %[[CURRENT_CHECK]], %[[START]]
// LLVM:   br i1 %[[DONE]], label %[[LOOP]], label %[[EXIT:.*]]
// LLVM: [[LOOP]]:
// LLVM:   %[[CURRENT:.*]] = load ptr, ptr %[[ITER]]
// LLVM:   call void @_ZN1SD1Ev(ptr %[[CURRENT]])
// LLVM:   %[[NEXT:.*]] = getelementptr %struct.S, ptr %[[CURRENT]], i64 -1
// LLVM:   store ptr %[[NEXT]], ptr %[[ITER]]
// LLVM:   br label %[[COND]]
// LLVM: [[EXIT]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z18test_cleanup_arrayv()
// OGCG:   %[[ARRAY:.*]] = alloca [42 x %struct.S]
// OGCG:   %[[START:.*]] = getelementptr{{.*}} %struct.S{{.*}}
// OGCG:   %[[END:.*]] = getelementptr{{.*}} %struct.S{{.*}} i64 42
// OGCG:   br label %[[LOOP:.*]]
// OGCG: [[LOOP]]:
// OGCG:   %[[NEXT:.*]] = phi ptr [ %[[END]], %{{.*}} ], [ %[[LAST:.*]], %[[LOOP]] ]
// OGCG:   %[[LAST]] = getelementptr{{.*}} %struct.S{{.*}}, ptr %[[NEXT]], i64 -1
// OGCG:   call void @_ZN1SD1Ev(ptr{{.*}} %[[LAST]])
// OGCG:   %[[DONE:.*]] = icmp eq ptr %[[LAST]], %[[START]]
// OGCG:   br i1 %[[DONE]], label %[[EXIT:.*]], label %[[LOOP]]
// OGCG: [[EXIT]]:
// OGCG:   ret void

void test_cleanup_zero_length_array() {
    S s[0];
}

// CIR:     cir.func{{.*}} @_Z30test_cleanup_zero_length_arrayv()
// CIR:       %[[S:.*]] = cir.alloca !cir.array<!rec_S x 0>, !cir.ptr<!cir.array<!rec_S x 0>>, ["s"]
// CIR-NOT:   cir.array.dtor
// CIR:       cir.return

// LLVM:     define{{.*}} void @_Z30test_cleanup_zero_length_arrayv()
// LLVM:       alloca [0 x %struct.S]
// LLVM-NOT:   call void @_ZN1SD1Ev
// LLVM:       ret void

// OGCG:     define{{.*}} void @_Z30test_cleanup_zero_length_arrayv()
// OGCG:       alloca [0 x %struct.S]
// OGCG-NOT:   call void @_ZN1SD1Ev
// OGCG:       ret void

