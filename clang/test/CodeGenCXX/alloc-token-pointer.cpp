// RUN: %clang_cc1 -fsanitize=alloc-token -triple x86_64-linux-gnu -std=c++20 -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s

#include "../Analysis/Inputs/system-header-simulator-cxx.h"

typedef __UINTPTR_TYPE__ uintptr_t;

extern "C" {
void *malloc(size_t size);
}

// CHECK-LABEL: define dso_local noundef ptr @_Z15test_malloc_intv(
// CHECK: call ptr @malloc(i64 noundef 4)
void *test_malloc_int() {
  int *a = (int *)malloc(sizeof(int));
  *a = 42;
  return a;
}

// CHECK-LABEL: define dso_local noundef ptr @_Z15test_malloc_ptrv(
// CHECK: call ptr @malloc(i64 noundef 8)
int **test_malloc_ptr() {
  int **a = (int **)malloc(sizeof(int*));
  *a = nullptr;
  return a;
}

// CHECK-LABEL: define dso_local noundef ptr @_Z12test_new_intv(
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 4){{.*}} !alloc_token [[META_INT:![0-9]+]]
int *test_new_int() {
  return new int;
}

// CHECK-LABEL: define dso_local noundef ptr @_Z20test_new_ulong_arrayv(
// CHECK: call noalias noundef nonnull ptr @_Znam(i64 noundef 80){{.*}} !alloc_token [[META_ULONG:![0-9]+]]
unsigned long *test_new_ulong_array() {
  return new unsigned long[10];
}

// CHECK-LABEL: define dso_local noundef ptr @_Z12test_new_ptrv(
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 8){{.*}} !alloc_token [[META_INTPTR:![0-9]+]]
int **test_new_ptr() {
  return new int*;
}

// CHECK-LABEL: define dso_local noundef ptr @_Z18test_new_ptr_arrayv(
// CHECK: call noalias noundef nonnull ptr @_Znam(i64 noundef 80){{.*}} !alloc_token [[META_INTPTR]]
int **test_new_ptr_array() {
  return new int*[10];
}

struct ContainsPtr {
  int a;
  char *buf;
};

// CHECK-LABEL: define dso_local noundef ptr @_Z27test_malloc_struct_with_ptrv(
// CHECK: call ptr @malloc(i64 noundef 16)
ContainsPtr *test_malloc_struct_with_ptr() {
  ContainsPtr *c = (ContainsPtr *)malloc(sizeof(ContainsPtr));
  return c;
}

// CHECK-LABEL: define dso_local noundef ptr @_Z33test_malloc_struct_array_with_ptrv(
// CHECK: call ptr @malloc(i64 noundef 160)
ContainsPtr *test_malloc_struct_array_with_ptr() {
  ContainsPtr *c = (ContainsPtr *)malloc(10 * sizeof(ContainsPtr));
  return c;
}

// CHECK-LABEL: define dso_local noundef ptr @_Z32test_operatornew_struct_with_ptrv(
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 16)
ContainsPtr *test_operatornew_struct_with_ptr() {
  ContainsPtr *c = (ContainsPtr *)__builtin_operator_new(sizeof(ContainsPtr));
  return c;
}

// CHECK-LABEL: define dso_local noundef ptr @_Z38test_operatornew_struct_array_with_ptrv(
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 160)
ContainsPtr *test_operatornew_struct_array_with_ptr() {
  ContainsPtr *c = (ContainsPtr *)__builtin_operator_new(10 * sizeof(ContainsPtr));
  return c;
}

// CHECK-LABEL: define dso_local noundef ptr @_Z33test_operatornew_struct_with_ptr2v(
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 16)
ContainsPtr *test_operatornew_struct_with_ptr2() {
  ContainsPtr *c = (ContainsPtr *)__builtin_operator_new(sizeof(*c));
  return c;
}

// CHECK-LABEL: define dso_local noundef ptr @_Z39test_operatornew_struct_array_with_ptr2v(
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 160)
ContainsPtr *test_operatornew_struct_array_with_ptr2() {
  ContainsPtr *c = (ContainsPtr *)__builtin_operator_new(10 * sizeof(*c));
  return c;
}

// CHECK-LABEL: define dso_local noundef ptr @_Z24test_new_struct_with_ptrv(
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 16){{.*}} !alloc_token [[META_CONTAINSPTR:![0-9]+]]
ContainsPtr *test_new_struct_with_ptr() {
  return new ContainsPtr;
}

// CHECK-LABEL: define dso_local noundef ptr @_Z30test_new_struct_array_with_ptrv(
// CHECK: call noalias noundef nonnull ptr @_Znam(i64 noundef 160){{.*}} !alloc_token [[META_CONTAINSPTR]]
ContainsPtr *test_new_struct_array_with_ptr() {
  return new ContainsPtr[10];
}

class TestClass {
public:
  void Foo();
  ~TestClass();
  int data[16];
};

// CHECK-LABEL: define dso_local noundef ptr @_Z14test_new_classv(
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 64){{.*}} !alloc_token [[META_TESTCLASS:![0-9]+]]
TestClass *test_new_class() {
  return new TestClass();
}

// CHECK-LABEL: define dso_local noundef ptr @_Z20test_new_class_arrayv(
// CHECK: call noalias noundef nonnull ptr @_Znam(i64 noundef 648){{.*}} !alloc_token [[META_TESTCLASS]]
TestClass *test_new_class_array() {
  return new TestClass[10];
}

// Test that we detect that virtual classes have implicit vtable pointer.
class VirtualTestClass {
public:
  virtual void Foo();
  virtual ~VirtualTestClass();
  int data[16];
};

// CHECK-LABEL: define dso_local noundef ptr @_Z22test_new_virtual_classv(
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 72){{.*}} !alloc_token [[META_VIRTUALTESTCLASS:![0-9]+]]
VirtualTestClass *test_new_virtual_class() {
  return new VirtualTestClass();
}

// CHECK-LABEL: define dso_local noundef ptr @_Z28test_new_virtual_class_arrayv(
// CHECK: call noalias noundef nonnull ptr @_Znam(i64 noundef 728){{.*}} !alloc_token [[META_VIRTUALTESTCLASS]]
VirtualTestClass *test_new_virtual_class_array() {
  return new VirtualTestClass[10];
}

// uintptr_t is treated as a pointer.
struct MyStructUintptr {
  int a;
  uintptr_t ptr;
};

// CHECK-LABEL: define dso_local noundef ptr @_Z18test_uintptr_isptrv(
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 16){{.*}} !alloc_token [[META_MYSTRUCTUINTPTR:![0-9]+]]
MyStructUintptr *test_uintptr_isptr() {
  return new MyStructUintptr;
}

using uptr = uintptr_t;
// CHECK-LABEL: define dso_local noundef ptr @_Z19test_uintptr_isptr2v(
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 8){{.*}} !alloc_token [[META_UINTPTR:![0-9]+]]
uptr *test_uintptr_isptr2() {
  return new uptr;
}

// CHECK: [[META_INT]] = !{!"int", i1 false}
// CHECK: [[META_ULONG]] = !{!"unsigned long", i1 false}
// CHECK: [[META_INTPTR]] = !{!"int *", i1 true}
// CHECK: [[META_CONTAINSPTR]] = !{!"ContainsPtr", i1 true}
// CHECK: [[META_TESTCLASS]] = !{!"TestClass", i1 false}
// CHECK: [[META_VIRTUALTESTCLASS]] = !{!"VirtualTestClass", i1 true}
// CHECK: [[META_MYSTRUCTUINTPTR]] = !{!"MyStructUintptr", i1 true}
// CHECK: [[META_UINTPTR]] = !{!"unsigned long", i1 true}
