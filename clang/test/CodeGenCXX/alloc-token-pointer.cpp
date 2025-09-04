// Check -fsanitize=alloc-token TypeHashPointerSplit mode with only 2
// tokens so we effectively only test the contains-pointer logic.
//
// RUN: %clang_cc1    -fsanitize=alloc-token -falloc-token-max=2 -triple x86_64-linux-gnu -std=c++20 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O -fsanitize=alloc-token -falloc-token-max=2 -triple x86_64-linux-gnu -std=c++20 -emit-llvm %s -o - | FileCheck %s

#include "../Analysis/Inputs/system-header-simulator-cxx.h"

typedef __UINTPTR_TYPE__ uintptr_t;

extern "C" {
void *malloc(size_t size);
}

// CHECK-LABEL: @_Z15test_malloc_intv(
void *test_malloc_int() {
  // CHECK: call{{.*}} ptr @__alloc_token_malloc(i64 noundef 4, i64 0)
  int *a = (int *)malloc(sizeof(int));
  *a = 42;
  return a;
}

// CHECK-LABEL: @_Z15test_malloc_ptrv(
int **test_malloc_ptr() {
  // FIXME: This should not be token ID 0!
  // CHECK: call{{.*}} ptr @__alloc_token_malloc(i64 noundef 8, i64 0)
  int **a = (int **)malloc(sizeof(int*));
  *a = nullptr;
  return a;
}

// CHECK-LABEL: @_Z12test_new_intv(
int *test_new_int() {
  // CHECK: call {{.*}} ptr @__alloc_token_Znwm(i64 noundef 4, i64 0){{.*}} !alloc_token_hint
  return new int;
}

// CHECK-LABEL: @_Z20test_new_ulong_arrayv(
unsigned long *test_new_ulong_array() {
  // CHECK: call {{.*}} ptr @__alloc_token_Znam(i64 noundef 80, i64 0){{.*}} !alloc_token_hint
  return new unsigned long[10];
}

// CHECK-LABEL: @_Z12test_new_ptrv(
int **test_new_ptr() {
  // CHECK: call {{.*}} ptr @__alloc_token_Znwm(i64 noundef 8, i64 1){{.*}} !alloc_token_hint
  return new int*;
}

// CHECK-LABEL: @_Z18test_new_ptr_arrayv(
int **test_new_ptr_array() {
  // CHECK: call {{.*}} ptr @__alloc_token_Znam(i64 noundef 80, i64 1){{.*}} !alloc_token_hint
  return new int*[10];
}

struct ContainsPtr {
  int a;
  char *buf;
};

// CHECK-LABEL: @_Z27test_malloc_struct_with_ptrv(
ContainsPtr *test_malloc_struct_with_ptr() {
  // FIXME: This should not be token ID 0!
  // CHECK: call{{.*}} ptr @__alloc_token_malloc(i64 noundef 16, i64 0)
  ContainsPtr *c = (ContainsPtr *)malloc(sizeof(ContainsPtr));
  return c;
}

// CHECK-LABEL: @_Z33test_malloc_struct_array_with_ptrv(
ContainsPtr *test_malloc_struct_array_with_ptr() {
  // FIXME: This should not be token ID 0!
  // CHECK: call{{.*}} ptr @__alloc_token_malloc(i64 noundef 160, i64 0)
  ContainsPtr *c = (ContainsPtr *)malloc(10 * sizeof(ContainsPtr));
  return c;
}

// CHECK-LABEL: @_Z32test_operatornew_struct_with_ptrv(
ContainsPtr *test_operatornew_struct_with_ptr() {
  // FIXME: This should not be token ID 0!
  // CHECK: call {{.*}} ptr @__alloc_token_Znwm(i64 noundef 16, i64 0)
  ContainsPtr *c = (ContainsPtr *)__builtin_operator_new(sizeof(ContainsPtr));
  return c;
}

// CHECK-LABEL: @_Z38test_operatornew_struct_array_with_ptrv(
ContainsPtr *test_operatornew_struct_array_with_ptr() {
  // FIXME: This should not be token ID 0!
  // CHECK: call {{.*}} ptr @__alloc_token_Znwm(i64 noundef 160, i64 0)
  ContainsPtr *c = (ContainsPtr *)__builtin_operator_new(10 * sizeof(ContainsPtr));
  return c;
}

// CHECK-LABEL: @_Z33test_operatornew_struct_with_ptr2v(
ContainsPtr *test_operatornew_struct_with_ptr2() {
  // FIXME: This should not be token ID 0!
  // CHECK: call {{.*}} ptr @__alloc_token_Znwm(i64 noundef 16, i64 0)
  ContainsPtr *c = (ContainsPtr *)__builtin_operator_new(sizeof(*c));
  return c;
}

// CHECK-LABEL: @_Z39test_operatornew_struct_array_with_ptr2v(
ContainsPtr *test_operatornew_struct_array_with_ptr2() {
  // FIXME: This should not be token ID 0!
  // CHECK: call {{.*}} ptr @__alloc_token_Znwm(i64 noundef 160, i64 0)
  ContainsPtr *c = (ContainsPtr *)__builtin_operator_new(10 * sizeof(*c));
  return c;
}

// CHECK-LABEL: @_Z24test_new_struct_with_ptrv(
ContainsPtr *test_new_struct_with_ptr() {
  // CHECK: call {{.*}} ptr @__alloc_token_Znwm(i64 noundef 16, i64 1){{.*}} !alloc_token_hint
  return new ContainsPtr;
}

// CHECK-LABEL: @_Z30test_new_struct_array_with_ptrv(
ContainsPtr *test_new_struct_array_with_ptr() {
  // CHECK: call {{.*}} ptr @__alloc_token_Znam(i64 noundef 160, i64 1){{.*}} !alloc_token_hint
  return new ContainsPtr[10];
}

class TestClass {
public:
  void Foo();
  ~TestClass();
  int data[16];
};

// CHECK-LABEL: @_Z14test_new_classv(
TestClass *test_new_class() {
  // CHECK: call {{.*}} ptr @__alloc_token_Znwm(i64 noundef 64, i64 0){{.*}} !alloc_token_hint
  return new TestClass();
}

// CHECK-LABEL: @_Z20test_new_class_arrayv(
TestClass *test_new_class_array() {
  // CHECK: call {{.*}} ptr @__alloc_token_Znam(i64 noundef 648, i64 0){{.*}} !alloc_token_hint
  return new TestClass[10];
}

// Test that we detect that virtual classes have implicit vtable pointer.
class VirtualTestClass {
public:
  virtual void Foo();
  virtual ~VirtualTestClass();
  int data[16];
};

// CHECK-LABEL: @_Z22test_new_virtual_classv(
VirtualTestClass *test_new_virtual_class() {
  // CHECK: call {{.*}} ptr @__alloc_token_Znwm(i64 noundef 72, i64 1){{.*}} !alloc_token_hint
  return new VirtualTestClass();
}

// CHECK-LABEL: @_Z28test_new_virtual_class_arrayv(
VirtualTestClass *test_new_virtual_class_array() {
  // CHECK: call {{.*}} ptr @__alloc_token_Znam(i64 noundef 728, i64 1){{.*}} !alloc_token_hint
  return new VirtualTestClass[10];
}

// uintptr_t is treated as a pointer.
struct MyStructUintptr {
  int a;
  uintptr_t ptr;
};

// CHECK-LABEL: @_Z18test_uintptr_isptrv(
MyStructUintptr *test_uintptr_isptr() {
  // CHECK: call {{.*}} ptr @__alloc_token_Znwm(i64 noundef 16, i64 1)
  return new MyStructUintptr;
}

using uptr = uintptr_t;
// CHECK-LABEL: @_Z19test_uintptr_isptr2v(
uptr *test_uintptr_isptr2() {
  // CHECK: call {{.*}} ptr @__alloc_token_Znwm(i64 noundef 8, i64 1)
  return new uptr;
}
