// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct A {
  A();
  virtual void f(char);
};

// This should initialize the vtable pointer.
A::A() {}

// CIR: !rec_A = !cir.record<struct "A" {!cir.vptr}>
// CIR: !rec_anon_struct = !cir.record<struct  {!cir.array<!cir.ptr<!u8i> x 3>}>

// CIR: cir.global "private" external @_ZTV1A : !rec_anon_struct

// LLVM: @_ZTV1A = external global { [3 x ptr] }

// OGCG: @_ZTV1A = external unnamed_addr constant { [3 x ptr] }

// CIR: cir.func{{.*}} @_ZN1AC2Ev(%arg0: !cir.ptr<!rec_A> {{.*}})
// CIR:    %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["this", init]
// CIR:    cir.store %arg0, %[[THIS_ADDR]] : !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>
// CIR:    %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
// CIR:    %[[VPTR:.*]] = cir.vtable.address_point(@_ZTV1A, address_point = <index = 0, offset = 2>) : !cir.vptr
// CIR:    %[[THIS_VPTR_PTR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_A> -> !cir.ptr<!cir.vptr>
// CIR:    cir.store{{.*}} align(8) %[[VPTR]], %[[THIS_VPTR_PTR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR:    cir.return

// LLVM: define{{.*}} void @_ZN1AC2Ev(ptr %[[ARG0:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[ARG0]], ptr %[[THIS_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1A, i64 16), ptr %[[THIS]]

// OGCG: define{{.*}} void @_ZN1AC2Ev(ptr {{.*}} %[[ARG0:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[ARG0]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 2), ptr %[[THIS]]

// NOTE: The GEP in OGCG looks very different from the one generated with CIR,
//       but it is equivalent. The OGCG GEP indexes by base pointer, then
//       structure, then array, whereas the CIR GEP indexes by byte offset.

void f1(A *a) {
  a->f('c');
}

// CIR: cir.func{{.*}} @_Z2f1P1A(%arg0: !cir.ptr<!rec_A> {{.*}})
// CIR:   %[[A_ADDR:.*]] = cir.alloca !cir.ptr<!rec_A>
// CIR:   cir.store %arg0, %[[A_ADDR]]
// CIR:   %[[A:.*]] = cir.load{{.*}} %[[A_ADDR]]
// CIR:   %[[C_LITERAL:.*]] = cir.const #cir.int<99> : !s8i
// CIR:   %[[VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[A]] : !cir.ptr<!rec_A> -> !cir.ptr<!cir.vptr>
// CIR:   %[[VPTR:.*]] = cir.load{{.*}} %[[VPTR_ADDR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:   %[[FN_PTR_PTR:.*]] = cir.vtable.get_virtual_fn_addr %[[VPTR]][0] : !cir.vptr -> !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_A>, !s8i)>>>
// CIR:   %[[FN_PTR:.*]] = cir.load{{.*}} %[[FN_PTR_PTR:.*]] : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_A>, !s8i)>>>, !cir.ptr<!cir.func<(!cir.ptr<!rec_A>, !s8i)>>
// CIR:   cir.call %[[FN_PTR]](%[[A]], %[[C_LITERAL]]) : (!cir.ptr<!cir.func<(!cir.ptr<!rec_A>, !s8i)>>, !cir.ptr<!rec_A>, !s8i) -> ()

// LLVM: define{{.*}} void @_Z2f1P1A(ptr %[[ARG0:.*]])
// LLVM:   %[[A_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[ARG0]], ptr %[[A_ADDR]]
// LLVM:   %[[A:.*]] = load ptr, ptr %[[A_ADDR]]
// LLVM:   %[[VPTR:.*]] = load ptr, ptr %[[A]]
// LLVM:   %[[FN_PTR_PTR:.*]] = getelementptr inbounds ptr, ptr %[[VPTR]], i32 0
// LLVM:   %[[FN_PTR:.*]] = load ptr, ptr %[[FN_PTR_PTR]]
// LLVM:   call void %[[FN_PTR]](ptr %[[A]], i8 99)

// OGCG: define{{.*}} void @_Z2f1P1A(ptr {{.*}} %[[ARG0:.*]])
// OGCG:   %[[A_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[ARG0]], ptr %[[A_ADDR]]
// OGCG:   %[[A:.*]] = load ptr, ptr %[[A_ADDR]]
// OGCG:   %[[VPTR:.*]] = load ptr, ptr %[[A]]
// OGCG:   %[[FN_PTR_PTR:.*]] = getelementptr inbounds ptr, ptr %[[VPTR]], i64 0
// OGCG:   %[[FN_PTR:.*]] = load ptr, ptr %[[FN_PTR_PTR]]
// OGCG:   call void %[[FN_PTR]](ptr {{.*}} %[[A]], i8 {{.*}} 99)
