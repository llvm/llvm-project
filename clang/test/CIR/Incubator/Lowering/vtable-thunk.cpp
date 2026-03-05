// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ogcg.ll %s

// Test that thunks lower correctly from CIR to LLVM IR and match OGCG output

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
// VTable Structure Validation
// ============================================================================

// Check vtable contains thunk with correct offset (16 bytes on x86_64)
// Both CIR and OGCG should produce identical vtable structure
// LLVM: @_ZTV7Derived = linkonce_odr constant { [4 x ptr], [3 x ptr] }
// LLVM-SAME: @_ZThn16_N7Derived3barEv

// OGCG: @_ZTV7Derived = linkonce_odr {{.*}} constant { [4 x ptr], [3 x ptr] }
// OGCG-SAME: @_ZThn16_N7Derived3barEv

// ============================================================================
// Thunk Implementation - This Pointer Adjustment
// ============================================================================

// CIR lowering should produce the same pointer adjustment as OGCG
// LLVM-LABEL: define linkonce_odr void @_ZThn16_N7Derived3barEv
// LLVM: %[[VAR1:[0-9]+]] = getelementptr i8, ptr %{{[0-9]+}}, i64 -16
// LLVM: call void @_ZN7Derived3barEv(ptr %[[VAR1]])

// OGCG-LABEL: define linkonce_odr void @_ZThn16_N7Derived3barEv
// OGCG: %[[VAR2:[0-9]+]] = getelementptr inbounds i8, ptr %{{.*}}, i64 -16
// OGCG: call void @_ZN7Derived3barEv(ptr {{.*}} %[[VAR2]])

