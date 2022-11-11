// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -emit-llvm < %s | FileCheck %s

_Static_assert(sizeof(void *) == 8, "sizeof(void *) has unexpected value.  Expected 8.");

int foo(void) {
  // CHECK: define dso_local i32 @foo
  // CHECK: %a = alloca ptr addrspace(270), align 4
  // CHECK: ret i32 4
  int (*__ptr32 a)(int);
  return sizeof(a);
}

int bar(void) {
  // CHECK: define dso_local i32 @bar
  // CHECK: %p = alloca ptr addrspace(270), align 4
  // CHECK: ret i32 4
  int *__ptr32 p;
  return sizeof(p);
}


int baz(void) {
  // CHECK: define dso_local i32 @baz
  // CHECK: %p = alloca ptr addrspace(270), align 4
  // CHECK: ret i32 4
  typedef int *__ptr32 IP32_PTR;

  IP32_PTR p;
  return sizeof(p);
}

int fugu(void) {
  // CHECK: define dso_local i32 @fugu
  // CHECK: %p = alloca ptr addrspace(270), align 4
  // CHECK: ret i32 4
  typedef int *int_star;

  int_star __ptr32 p;
  return sizeof(p);
}
