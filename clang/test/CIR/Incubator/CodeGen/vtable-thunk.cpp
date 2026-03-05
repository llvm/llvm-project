// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Test basic thunk generation for multiple inheritance with non-virtual thunks

class Base1 {
public:
  virtual void foo() {}
  int x;
};

class Base2 {
public:
  virtual void bar() {}
  int y;
};

class Derived : public Base1, public Base2 {
public:
  void bar() override {}
};

void test() {
  Derived d;
  Base2* b2 = &d;
  b2->bar();
}

// ============================================================================
// CIR VTable Structure
// ============================================================================

// Check thunk is in vtable
// CIR: cir.global constant linkonce_odr @_ZTV7Derived = #cir.vtable
// CIR: #cir.global_view<@_ZThn16_N7Derived3barEv>

// ============================================================================
// CIR Thunk Function Generation
// ============================================================================

// Check that thunk function is generated with:
// - comdat attribute (for deduplication across TUs)
// - linkonce_odr linkage (one definition rule, discardable)
// - correct mangling (_ZThn<offset>_<original_name>)
// CIR: cir.func {{.*}}comdat linkonce_odr @_ZThn16_N7Derived3barEv

// ============================================================================
// CIR Thunk Implementation - This Pointer Adjustment
// ============================================================================

// The thunk should:
// 1. Adjust the 'this' pointer by the offset (-16 bytes)
// 2. Call the actual implementation with the adjusted pointer

// CIR: cir.ptr_stride
// CIR: cir.call @_ZN7Derived3barEv

// ============================================================================
// LLVM IR Output Validation
// ============================================================================

//      LLVM: @_ZTV7Derived = linkonce_odr constant
// LLVM-SAME: @_ZThn16_N7Derived3barEv

//      LLVM: define linkonce_odr void @_ZThn16_N7Derived3barEv
// LLVM-SAME: ptr

// ============================================================================
// Test Multiple Base Classes (Different Offsets)
// ============================================================================

class A {
public:
  virtual void methodA() {}
  long long a;  // 8 bytes
};

class B {
public:
  virtual void methodB() {}
  long long b;  // 8 bytes
};

class C {
public:
  virtual void methodC() {}
  long long c;  // 8 bytes
};

class Multi : public A, public B, public C {
public:
  void methodB() override {}
  void methodC() override {}
};

void test_multi() {
  Multi m;
  B* pb = &m;
  C* pc = &m;
  pb->methodB();
  pc->methodC();
}

// Different thunks for different offsets
// Offset to B should be 16 (A's vptr + a)
// CIR: cir.func {{.*}}comdat linkonce_odr @_ZThn16_N5Multi7methodBEv

// Offset to C should be 32 (A's vptr + a + B's vptr + b)
// CIR: cir.func {{.*}}comdat linkonce_odr @_ZThn32_N5Multi7methodCEv
