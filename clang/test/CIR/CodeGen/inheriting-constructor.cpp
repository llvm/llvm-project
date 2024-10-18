// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR_ITANIUM --input-file=%t.cir %s

struct A { A(int); virtual ~A(); };
struct B : A { using A::A; ~B(); };
B::~B() {}

B b(123);

// CIR_ITANIUM-LABEL: @_ZN1BD2Ev
// CIR_ITANIUM-LABEL: @_ZN1BD1Ev
// CIR_ITANIUM-LABEL: @_ZN1BD0Ev