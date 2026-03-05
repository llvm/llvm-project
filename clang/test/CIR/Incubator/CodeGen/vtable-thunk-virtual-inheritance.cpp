// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ogcg.ll %s

// Test thunk generation with virtual inheritance (diamond problem)

class Base {
public:
  virtual void method() {}
  int a;
};

class Left : public virtual Base {
public:
  virtual void leftMethod() {}
  int b;
};

class Right : public virtual Base {
public:
  virtual void rightMethod() {}
  int c;
};

class Diamond : public Left, public Right {
public:
  void leftMethod() override {}
  void rightMethod() override {}
};

void test() {
  Diamond d;
  Left* l = &d;
  Right* r = &d;
  l->leftMethod();
  r->rightMethod();
}

// ============================================================================
// CIR Output - Thunk Generation
// ============================================================================

// Diamond's rightMethod needs a thunk because Right is at offset 16
// leftMethod doesn't need a thunk because Left is at offset 0
// CIR: cir.func {{.*}}comdat linkonce_odr @_ZThn16_N7Diamond11rightMethodEv
// CIR: cir.ptr_stride
// CIR: cir.call @_ZN7Diamond11rightMethodEv

// ============================================================================
// VTable Structure - Both CIR and OGCG
// ============================================================================

// Check that vtable contains the thunk reference at the correct position
//      LLVM: @_ZTV7Diamond = linkonce_odr constant
// LLVM-SAME: @_ZThn16_N7Diamond11rightMethodEv

//      OGCG: @_ZTV7Diamond = linkonce_odr {{.*}} constant
// OGCG-SAME: @_ZThn16_N7Diamond11rightMethodEv

// ============================================================================
// Thunk Implementation - LLVM Lowering vs OGCG
// ============================================================================

// CIR lowering should produce the same this-pointer adjustment as OGCG
// LLVM-LABEL: define linkonce_odr void @_ZThn16_N7Diamond11rightMethodEv
//      LLVM: %[[VAR1:[0-9]+]] = getelementptr i8, ptr %{{[0-9]+}}, i64 -16
//      LLVM: call void @_ZN7Diamond11rightMethodEv(ptr %[[VAR1]])

// OGCG-LABEL: define linkonce_odr void @_ZThn16_N7Diamond11rightMethodEv
//      OGCG: %[[VAR2:[0-9]+]] = getelementptr inbounds i8, ptr %{{.*}}, i64 -16
//      OGCG: call void @_ZN7Diamond11rightMethodEv(ptr {{.*}} %[[VAR2]])
