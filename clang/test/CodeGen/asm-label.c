// RUN: %clang_cc1 -triple=i686-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LINUX
// RUN: %clang_cc1 -triple=i686-apple-darwin9 -emit-llvm %s -o - | FileCheck %s --check-prefix=DARWIN

char *strerror(int) asm("alias");
int x __asm("foo");

int *test(void) {
  static int y __asm("bar");
  strerror(-1);
  return &y;
}

// LINUX: @bar = internal global i32 0
// LINUX: @foo = common global i32 0
// LINUX: declare i8* @alias(i32)

// DARWIN: @"\01bar" = internal global i32 0
// DARWIN: @"\01foo" = common global i32 0
// DARWIN: declare i8* @"\01alias"(i32)
