// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ogcg.ll %s

// Test thunk generation with multiple base classes
// This validates thunks for void-returning methods (no return adjustment).
// Full covariant return adjustment for pointer-returning methods is NYI.

class Base1 {
public:
  virtual void foo() {}
};

class Base2 {
public:
  virtual void bar() {}
  int data;
};

class Derived : public Base1, public Base2 {
public:
  void bar() override {}
};

void test() {
  Derived d;
  Base2* b2 = &d;
  b2->bar();  // Needs this-adjusting thunk (no return adjustment)
}

// ============================================================================
// CIR Output - Thunk with This-Adjustment Only
// ============================================================================

// Derived::bar() needs a thunk when called through Base2* because
// Base2 is at offset 8 in Derived (after Base1's vtable pointer)

// CIR: cir.func {{.*}}comdat linkonce_odr @_ZThn8_N7Derived3barEv
// CIR: cir.ptr_stride
// CIR: cir.call @_ZN7Derived3barEv

// ============================================================================
// VTable Structure - Both CIR and OGCG
// ============================================================================

// Check that vtable contains the thunk
//      LLVM: @_ZTV7Derived = linkonce_odr constant
// LLVM-SAME: @_ZThn8_N7Derived3barEv

//      OGCG: @_ZTV7Derived = linkonce_odr {{.*}} constant
// OGCG-SAME: @_ZThn8_N7Derived3barEv

// ============================================================================
// Thunk Implementation - LLVM Lowering vs OGCG
// ============================================================================

// CIR lowering should produce this-adjustment (no return adjustment for void)
// LLVM-LABEL: define linkonce_odr void @_ZThn8_N7Derived3barEv
//       LLVM: getelementptr i8, ptr %{{[0-9]+}}, i64 -8
//       LLVM: call void @_ZN7Derived3barEv

// OGCG-LABEL: define linkonce_odr void @_ZThn8_N7Derived3barEv
//       OGCG: getelementptr inbounds i8, ptr %{{.*}}, i64 -8
//       OGCG: call void @_ZN7Derived3barEv
