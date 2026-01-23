// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-unknown %s -emit-llvm -o - | FileCheck %s -check-prefix=CXX26
// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero %s -emit-llvm -o - | FileCheck %s -check-prefix=ZERO
// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=pattern %s -emit-llvm -o - | FileCheck %s -check-prefix=PATTERN

// Test for C++26 [[indeterminate]] attribute (P2795R5)
// The [[indeterminate]] attribute opts out of erroneous initialization.

template<typename T> void used(T &) noexcept;

extern "C" {

// Test: [[indeterminate]] should suppress zero/pattern initialization
// CXX26-LABEL: test_indeterminate_local_var(
// CXX26:       alloca i32
// CXX26-NOT:   store
// CXX26:       call void
// ZERO-LABEL:  test_indeterminate_local_var(
// ZERO:        alloca i32
// ZERO-NOT:    store
// ZERO:        call void
// PATTERN-LABEL: test_indeterminate_local_var(
// PATTERN:     alloca i32
// PATTERN-NOT: store
// PATTERN:     call void
void test_indeterminate_local_var() {
  [[indeterminate]] int x;
  used(x);
}

// Test: Without [[indeterminate]], zero/pattern init should apply
// CXX26-LABEL: test_normal_local_var(
// CXX26:       alloca i32
// CXX26-NEXT:  call void
// ZERO-LABEL:  test_normal_local_var(
// ZERO:        alloca i32
// ZERO:        store i32 0
// PATTERN-LABEL: test_normal_local_var(
// PATTERN:     alloca i32
// PATTERN:     store i32 -1431655766
void test_normal_local_var() {
  int y;
  used(y);
}

// Test: [[indeterminate]] on multiple variables
// ZERO-LABEL: test_indeterminate_multiple_vars(
// ZERO:       %a = alloca i32
// ZERO:       %b = alloca [10 x i32]
// ZERO:       %c = alloca [10 x [10 x i32]]
// ZERO-NOT:   call void @llvm.memset
// ZERO:       call void @_Z4used
void test_indeterminate_multiple_vars() {
  [[indeterminate]] int a, b[10], c[10][10];
  used(a);
}

// Test: Mixed indeterminate and normal variables
// ZERO-LABEL: test_mixed_vars(
void test_mixed_vars() {
  int normal = {};               // Explicitly zero-initialized
  [[indeterminate]] int indeterminate_var;
  int erroneous;                 // Will get erroneous initialization if -ftrivial-auto-var-init
  used(normal);
  used(indeterminate_var);
  used(erroneous);
}

} // extern "C"

// Test: Struct with indeterminate member initialization
struct SelfStorage {
  char data[512];
  void use_data();
};

// ZERO-LABEL: test_struct_indeterminate
// ZERO:       alloca %struct.SelfStorage
// ZERO-NOT:   call void @llvm.memset
void test_struct_indeterminate() {
  [[indeterminate]] SelfStorage s;
  s.use_data();
}
