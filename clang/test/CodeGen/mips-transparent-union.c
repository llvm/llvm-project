// RUN: %clang_cc1 -no-opaque-pointers -triple mips64-linux-gnu -S -o - -emit-llvm %s | FileCheck %s
//
// Transparent unions are passed according to the calling convention rules of
// the first member. In this case, it is as if it were a void pointer so we
// do not have the inreg attribute we would normally have for unions.
//
// This comes up in glibc's wait() function and matters for the big-endian N32
// case where pointers are promoted to i64 and a non-transparent union would be
// passed in the upper 32-bits of an i64.

union either_pointer {
  void *void_ptr;
  int *int_ptr;
} __attribute__((transparent_union));

extern void foo(union either_pointer p);

int data;

void bar(void) {
  return foo(&data);
}

// CHECK-LABEL: define{{.*}} void @bar()
// CHECK:         call void @foo(i8* %{{[0-9]+}})

// CHECK: declare void @foo(i8*)
