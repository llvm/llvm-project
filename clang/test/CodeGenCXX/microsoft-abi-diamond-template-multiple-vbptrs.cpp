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

// Test 1: Diamond inheritance in a template that triggers the bug
template<typename T>
class Derived : public Diamond {
public:
  // CHECK-LABEL: define {{.*}} @"??0?$Derived@H@test@@QEAA@XZ"

  // Layout of Derived<int>:
  // offset 0: vbptr for Left (8 bytes)
  // offset 8: vbptr for Right (8 bytes)
  // offset 16+: virtual base Base

  // CHECK: call {{.*}} @"??0Diamond@test@@QEAA@XZ"
  // EmitNullBaseClassInitialization now correctly calculates memory regions
  // around the vbptrs without hitting negative size assertion

  // No memset is generated since there are no gaps to zero
  // CHECK-NOT: call void @llvm.memset
  // CHECK: ret

  Derived() : Diamond() {}
};

// Explicit instantiation to trigger code generation
template class Derived<int>;

// Test 2: Diamond in a template, with data members (calls memset)
class DiamondWithData : public Left, public Right {
public:
  int x;
  int y;
};

template<typename T>
class DerivedWithData : public DiamondWithData {
public:
  DerivedWithData();
};

// CHECK-LABEL: define {{.*}} @"??0?$DerivedWithData@H@test@@QEAA@XZ"
//
// Layout of DerivedWithData<int>:
// offset 0: vbptr for Left (8 bytes)
// offset 8: vbptr for Right (8 bytes)
// offset 16: x (4 bytes)
// offset 20: y (4 bytes)
// offset 24+: virtual base Base
//
// EmitNullBaseClassInitialization zero-initializes the data members [16, 24)
// while skipping both vbptrs [0, 16)
//
// memset zeros 8 bytes for x and y at offset 16
// CHECK: call void @llvm.memset.p0.i64(ptr {{.*}}, i8 0, i64 8, i1 false)

template<typename T>
DerivedWithData<T>::DerivedWithData() : DiamondWithData() {
}

template struct DerivedWithData<int>;

// Test 3: Another case which triggers the bug (similar to Test 1, non-template)
class Interface {
public:
  virtual ~Interface() {}
};

class Base1If : public virtual Interface {
};

class Base2If : public virtual Interface {
};

class BaseIf : public Base1If, public Base2If {
};

class DerivedClass : public BaseIf {
public:
  // CHECK-LABEL: define {{.*}} @"??0DerivedClass@test@@QEAA@XZ"

  // Layout of DerivedClass:
  // offset 0: vbptr for Base1If (8 bytes)
  // offset 8: vbptr for Base2If (8 bytes)
  // offset 16+: virtual base Interface

  // CHECK: call {{.*}} @"??0BaseIf@test@@QEAA@XZ"
  // EmitNullBaseClassInitialization now correctly calculates memory regions
  // around the vbptrs without hitting negative size assertion

  // No memset is generated since there are no gaps to zero
  // CHECK-NOT: call void @llvm.memset
  // CHECK: ret

  DerivedClass()
  : BaseIf()
  { }
};

// Instantiate to trigger code generation
DerivedClass d;


} // namespace test
