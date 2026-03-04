// RUN: %clang_cc1 -triple x86_64-windows-msvc -std=c++11 -emit-llvm -o - %s | FileCheck %s

// Test for the fix in EmitNullBaseClassInitialization where the calculation
// of SplitAfterSize was incorrect when multiple vbptrs are present.

struct VBase1 {
  virtual ~VBase1();
  int x;
};

struct VBase2 {
  virtual ~VBase2();
  int y;
};

// Base class with two virtual base classes.
struct Base : virtual VBase1, virtual VBase2 {
  int data;
};

// Derived class that needs to initialize Base.
// The constructor will call EmitNullBaseClassInitialization for Base.
struct Derived : Base {
  Derived();
  int more_data;
};

// CHECK-LABEL: define dso_local noundef ptr @"??0Derived@@QEAA@XZ"
// Check that memory initialization (memset or memcpy) correctly covers
// the non-virtual portion of the base class, properly handling vbptrs.

// CHECK: call void @llvm.memset

Derived::Derived() : Base() {
  // Constructor body
  data = 42;
  more_data = 100;
}

// Test case with three virtual bases.
struct VBase3 {
  virtual ~VBase3();
  int z;
};

struct ComplexBase : virtual VBase1, virtual VBase2, virtual VBase3 {
  int a, b, c;
};

struct ComplexDerived : ComplexBase {
  ComplexDerived();
  int d;
};

// CHECK-LABEL: define dso_local noundef ptr @"??0ComplexDerived@@QEAA@XZ"
// CHECK: call void @llvm.memset

ComplexDerived::ComplexDerived() : ComplexBase() {
  a = 1;
  b = 2;
  c = 3;
  d = 4;
}

// Test case with data members initialized using memcpy
struct VBase4 {
  virtual ~VBase4();
  int w;
};

struct BaseWithPtrToMember : virtual VBase1, virtual VBase4 {
  int Base::*member_ptr;
  int value;
};

struct DerivedWithPtrToMember : BaseWithPtrToMember {
  DerivedWithPtrToMember();
};

// CHECK-LABEL: define dso_local noundef ptr @"??0DerivedWithPtrToMember@@QEAA@XZ"
// This should use memcpy instead of memset due to the pointer-to-member.
// CHECK: call void @llvm.memcpy

DerivedWithPtrToMember::DerivedWithPtrToMember() : BaseWithPtrToMember() {
  value = 50;
}
