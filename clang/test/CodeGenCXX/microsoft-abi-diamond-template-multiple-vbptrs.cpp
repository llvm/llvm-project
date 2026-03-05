// RUN: %clang_cc1 -triple x86_64-windows-msvc -std=c++17 -emit-llvm -o - %s | FileCheck %s

// Test for the fix in EmitNullBaseClassInitialization where the calculation
// of SplitAfterSize was incorrect when multiple vbptrs are present.

namespace test {

class Base {
public:
    virtual ~Base() {}
};

class Left : public virtual Base {
};

class Right : public virtual Base {
};

class Diamond : public Left, public Right {
};

// Template class that triggers the bug
template<typename T>
class Derived : public Diamond {
public:
    // CHECK-LABEL: define {{.*}} @"??0?$Derived@H@test@@QEAA@XZ"
    // CHECK: call {{.*}} @"??0Diamond@test@@QEAA@XZ"
    // EmitNullBaseClassInitialization now correctly calculates memory regions
    // around the vbptrs without hitting negative size assertion

    // CHECK: ret

    Derived() : Diamond() {}
};

// Explicit instantiation to trigger code generation
template class Derived<int>;

} // namespace test
