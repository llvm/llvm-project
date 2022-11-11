// RUN: %clang_cc1 %s -triple x86_64-pc-win32 -fms-extensions -emit-llvm -o - \
// RUN:         | FileCheck %s
// RUN: %clang_cc1 %s -triple i686-pc-win32 -fms-extensions -emit-llvm -o - \
// RUN:         | FileCheck %s
// RUN: %clang_cc1 %s -triple aarch64-windows -fms-extensions -emit-llvm -o - \
// RUN:         | FileCheck %s

// Check that the first finally block passes the enclosing function's frame
// pointer to the second finally block, instead of generating it via localaddr.

// CHECK-LABEL: define internal void @"?fin$0@0@main@@"({{i8 noundef( zeroext)?}} %abnormal_termination, ptr noundef %frame_pointer)
// CHECK: call void @"?fin$1@0@main@@"({{i8 noundef( zeroext)?}} 0, ptr noundef %frame_pointer)
int
main(void) {
  int Check = 0;
  __try {
    Check = 3;
  } __finally {
    __try {
      Check += 2;
    } __finally {
      Check += 4;
    }
  }
  return Check;
}
