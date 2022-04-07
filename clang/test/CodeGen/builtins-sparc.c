// REQUIRES: sparc-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple sparc-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -triple sparc64-unknown-unknown -emit-llvm %s -o - | FileCheck -check-prefix CHECK-V9 %s

void test_eh_return_data_regno(void)
{
  volatile int res;
  res = __builtin_eh_return_data_regno(0);  // CHECK,CHECKV9: store volatile i32 24
  res = __builtin_eh_return_data_regno(1);  // CHECK,CHECKV9: store volatile i32 25
}

void *test_extract_return_address(void)
{
  // CHECK,CHECKV9: getelementptr i8, i8* %0, i32 8
  return __builtin_extract_return_addr(__builtin_return_address(0));
}

struct s {
  void *p;
};

struct s test_extract_struct_return_address(void)
{
  struct s s;
  // CHECK:    getelementptr i8, i8* %0, i32 12
  // CHECK-V9: getelementptr i8, i8* %0, i32 8
  s.p = __builtin_extract_return_addr(__builtin_return_address(0));
  return s;
}
