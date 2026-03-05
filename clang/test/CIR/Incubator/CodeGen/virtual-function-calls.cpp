// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct A {
  virtual void f(char);
};

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
// LLVM:   %[[FN_PTR:.*]] = load ptr, ptr %5
// LLVM:   call void %[[FN_PTR]](ptr %[[A]], i8 99)

struct B : virtual A {
  virtual void f();
};

void f2(B * b) {
  b->f();
}

// CIR: cir.func{{.*}} @_Z2f2P1B(%arg0: !cir.ptr<!rec_B> {{.*}})
// CIR:   %[[B_ADDR:.*]] = cir.alloca !cir.ptr<!rec_B>
// CIR:   cir.store %arg0, %[[B_ADDR]]
// CIR:   %[[B:.*]] = cir.load{{.*}} %[[B_ADDR]]
// CIR:   %[[VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[B]] : !cir.ptr<!rec_B> -> !cir.ptr<!cir.vptr>
// CIR:   %[[VPTR:.*]] = cir.load{{.*}} %[[VPTR_ADDR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:   %[[FN_PTR_PTR:.*]] = cir.vtable.get_virtual_fn_addr %[[VPTR]][1] : !cir.vptr -> !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_B>)>>>
// CIR:   %[[FN_PTR:.*]] = cir.load{{.*}} %[[FN_PTR_PTR:.*]] : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_B>)>>>, !cir.ptr<!cir.func<(!cir.ptr<!rec_B>)>>
// CIR:   cir.call %[[FN_PTR]](%[[B]]) : (!cir.ptr<!cir.func<(!cir.ptr<!rec_B>)>>, !cir.ptr<!rec_B>) -> ()

// LLVM: define{{.*}} void @_Z2f2P1B(ptr %[[ARG0:.*]])
// LLVM:   %[[B_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[ARG0]], ptr %[[B_ADDR]]
// LLVM:   %[[B:.*]] = load ptr, ptr %[[B_ADDR]]
// LLVM:   %[[VPTR:.*]] = load ptr, ptr %[[B]]
// LLVM:   %[[FN_PTR_PTR:.*]] = getelementptr inbounds ptr, ptr %[[VPTR]], i32 1
// LLVM:   %[[FN_PTR:.*]] = load ptr, ptr %5
// LLVM:   call void %[[FN_PTR]](ptr %[[B]])
