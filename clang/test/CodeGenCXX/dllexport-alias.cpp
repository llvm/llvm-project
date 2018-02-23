// RUN: %clang_cc1 -triple x86_64-windows-gnu -mconstructor-aliases %s -S -emit-llvm -o - | FileCheck %s

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

// CHECK: @_ZN1AC1Ev = dso_local dllexport alias void (%class.A*), void (%class.A*)* @_ZN1AC2Ev
// CHECK: @_ZN1AD1Ev = dso_local dllexport alias void (%class.A*), void (%class.A*)* @_ZN1AD2Ev
