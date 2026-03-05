// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM

struct A1 {
  A1();
};

class B : public A1 {};

void f1() {
  B v{};
}

// CIR: cir.func {{.*}} @_Z2f1v()
// CIR:     %0 = cir.alloca !rec_B, !cir.ptr<!rec_B>, ["v", init]
// CIR:     %1 = cir.base_class_addr %0 : !cir.ptr<!rec_B> nonnull [0] -> !cir.ptr<!rec_A1>
// CIR:     cir.call @_ZN2A1C2Ev(%1) : (!cir.ptr<!rec_A1>) -> ()
// CIR:     cir.return
// LLVM: define dso_local void @_Z2f1v()
// LLVM:    %1 = alloca %class.B, i64 1, align 1
// LLVM:    call void @_ZN2A1C2Ev(ptr %1)
// LLVM:    ret void

struct A2 {
    A2();
};
class C : public A1, public A2 {};

void f2() {
  C v{};
}

// CIR: cir.func {{.*}} @_Z2f2v()
// CIR:     %0 = cir.alloca !rec_C, !cir.ptr<!rec_C>, ["v", init]
// CIR:     %1 = cir.base_class_addr %0 : !cir.ptr<!rec_C> nonnull [0] -> !cir.ptr<!rec_A1>
// CIR:     cir.call @_ZN2A1C2Ev(%1) : (!cir.ptr<!rec_A1>) -> ()
// CIR:     %2 = cir.base_class_addr %0 : !cir.ptr<!rec_C> nonnull [0] -> !cir.ptr<!rec_A2>
// CIR:     cir.call @_ZN2A2C2Ev(%2) : (!cir.ptr<!rec_A2>) -> ()
// CIR:     cir.return
// LLVM: define dso_local void @_Z2f2v()
// LLVM:    %1 = alloca %class.C, i64 1, align 1
// LLVM:    call void @_ZN2A1C2Ev(ptr %1)
// LLVM:    call void @_ZN2A2C2Ev(ptr %1)
// LLVM:    ret void

struct A3 {
    A3();
    ~A3();
};
class D : public A3 {};

void f3() {
  D v{};
}

// CIR: cir.func {{.*}} @_Z2f3v()
// CIR:     %0 = cir.alloca !rec_D, !cir.ptr<!rec_D>, ["v", init]
// CIR:     %1 = cir.base_class_addr %0 : !cir.ptr<!rec_D> nonnull [0] -> !cir.ptr<!rec_A3>
// CIR:     cir.call @_ZN2A3C2Ev(%1) : (!cir.ptr<!rec_A3>) -> ()
// CIR:     cir.call @_ZN1DD1Ev(%0) : (!cir.ptr<!rec_D>) -> ()
// CIR:     cir.return
// LLVM: define dso_local void @_Z2f3v()
// LLVM:    %1 = alloca %class.D, i64 1, align 1
// LLVM:    call void @_ZN2A3C2Ev(ptr %1)
// LLVM:    call void @_ZN1DD1Ev(ptr %1)
// LLVM:    ret void
