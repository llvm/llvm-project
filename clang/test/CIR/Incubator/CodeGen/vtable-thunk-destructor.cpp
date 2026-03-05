// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ogcg.ll %s

// Test thunk generation for virtual destructors in multiple inheritance

class Base1 {
public:
  virtual ~Base1() {}
  int x;
};

class Base2 {
public:
  virtual ~Base2() {}
  int y;
};

class Derived : public Base1, public Base2 {
public:
  ~Derived() override {}
};

void test() {
  Base2* b2 = new Derived();
  delete b2;  // Uses destructor thunk
}

// ============================================================================
// Destructor Thunks
// ============================================================================

// Derived's destructor needs thunks when called through Base2* because
// Base2 is at offset 16 in Derived (after Base1's vtable + data)
// The Itanium ABI generates multiple destructor variants:
// - D2 (base object destructor)
// - D1 (complete object destructor)
// - D0 (deleting destructor)

// Check for complete destructor thunk (D1) - appears first in output
// CIR: cir.func {{.*}}comdat linkonce_odr @_ZThn16_N7DerivedD1Ev
// CIR: cir.ptr_stride
// CIR: cir.call @_ZN7DerivedD1Ev

// Check for deleting destructor thunk (D0) - appears second in output
// CIR: cir.func {{.*}}comdat linkonce_odr @_ZThn16_N7DerivedD0Ev
// CIR: cir.ptr_stride
// CIR: cir.call @_ZN7DerivedD0Ev

// ============================================================================
// VTable Structure
// ============================================================================

// Check that vtable contains destructor thunks
//     LLVM: @_ZTV7Derived = linkonce_odr constant
// LLVM-DAG: @_ZThn16_N7DerivedD1Ev
// LLVM-DAG: @_ZThn16_N7DerivedD0Ev

//     OGCG: @_ZTV7Derived = linkonce_odr {{.*}} constant
// OGCG-DAG: @_ZThn16_N7DerivedD1Ev
// OGCG-DAG: @_ZThn16_N7DerivedD0Ev

// ============================================================================
// Thunk Implementation
// ============================================================================

// Complete destructor thunk (D1)
// LLVM-LABEL: define linkonce_odr void @_ZThn16_N7DerivedD1Ev
//       LLVM: getelementptr i8, ptr %{{[0-9]+}}, i64 -16
//       LLVM: call void @_ZN7DerivedD1Ev

// OGCG-LABEL: define linkonce_odr void @_ZThn16_N7DerivedD1Ev
//       OGCG: getelementptr inbounds i8, ptr %{{.*}}, i64 -16
//       OGCG: call void @_ZN7DerivedD1Ev

// Deleting destructor thunk (D0)
// LLVM-LABEL: define linkonce_odr void @_ZThn16_N7DerivedD0Ev
//       LLVM: getelementptr i8, ptr %{{[0-9]+}}, i64 -16
//       LLVM: call void @_ZN7DerivedD0Ev

// OGCG-LABEL: define linkonce_odr void @_ZThn16_N7DerivedD0Ev
//       OGCG: getelementptr inbounds i8, ptr %{{.*}}, i64 -16
//       OGCG: call void @_ZN7DerivedD0Ev
