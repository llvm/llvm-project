// RUN: %clang_cc1 -triple x86_64-windows-gnu -mconstructor-aliases %s -emit-llvm -o - | FileCheck %s

// This test assumes that the C1 constructor will be aliased to the C2
// constructor, and the D1 destructor to the D2. It then checks that the aliases
// are dllexport'ed.

class __declspec(dllexport) A {
public:
    A();
    ~A();
};

A::A() {}

A::~A() {}

// CHECK: @_ZN1AC1Ev = dso_local dllexport unnamed_addr alias void (ptr), ptr @_ZN1AC2Ev
// CHECK: @_ZN1AD1Ev = dso_local dllexport unnamed_addr alias void (ptr), ptr @_ZN1AD2Ev
