// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

class A {
    int a;
};

class B {
    int b;
public:
    A *getAsA();
};

class X : public A, public B {
    int x;
};

A *B::getAsA() {
  return static_cast<X*>(this);
}

// CIR-LABEL: @_ZN1B6getAsAEv
// CIR: %[[VAL_1:.*]] = cir.alloca !cir.ptr<!ty_B>, !cir.ptr<!cir.ptr<!ty_B>>, ["this", init] {alignment = 8 : i64}
// CIR: %[[VAL_2:.*]] = cir.alloca !cir.ptr<!ty_A>, !cir.ptr<!cir.ptr<!ty_A>>, ["__retval"] {alignment = 8 : i64}
// CIR: %[[VAL_3:.*]] = cir.load %[[VAL_1]] : !cir.ptr<!cir.ptr<!ty_B>>, !cir.ptr<!ty_B>
// CIR: %[[VAL_4:.*]] = cir.derived_class_addr(%[[VAL_3]] : !cir.ptr<!ty_B> nonnull) [4] -> !cir.ptr<!ty_X>
// CIR: %[[VAL_5:.*]] = cir.base_class_addr(%[[VAL_4]] : !cir.ptr<!ty_X>) [0] -> !cir.ptr<!ty_A>
// CIR: cir.store %[[VAL_5]], %[[VAL_2]] : !cir.ptr<!ty_A>, !cir.ptr<!cir.ptr<!ty_A>>
// CIR: %[[VAL_6:.*]] = cir.load %[[VAL_2]] : !cir.ptr<!cir.ptr<!ty_A>>, !cir.ptr<!ty_A>
// CIR: cir.return %[[VAL_6]] : !cir.ptr<!ty_A>

// LLVM-LABEL: @_ZN1B6getAsAEv
// LLVM:  %[[VAL_1:.*]] = alloca ptr, i64 1, align 8
// LLVM:  store ptr %[[VAL_2:.*]], ptr %[[VAL_0:.*]], align 8
// LLVM:  %[[VAL_3:.*]] = load ptr, ptr %[[VAL_0]], align 8
// LLVM:  %[[VAL_4:.*]] = getelementptr i8, ptr %[[VAL_3]], i32 -4
// LLVM-NOT: select i1
// LLVM:  ret ptr