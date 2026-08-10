// This test verifies that dispatch_sync() doesn't actually copy the block under TSan (without TSan, it doesn't).

// RUN: %clangxx_tsan %s -o %t_no_tsan -fno-sanitize=thread
// RUN: %clangxx_tsan %s -o %t_with_tsan

// RUN: %run %t_no_tsan   2>&1 | FileCheck %s
// RUN: %run %t_with_tsan 2>&1 | FileCheck %s --implicit-check-not='ThreadSanitizer'

#include <dispatch/dispatch.h>

#include <stdio.h>

struct MyClass {
  static int copyCount;
  static void printCopyCount() {
    fprintf(stderr, "copyCount = %d\n", copyCount);
  }
  MyClass(){};
  MyClass(const MyClass &obj) { copyCount++; };
  void foo() const {
    fprintf(stderr, "MyClass::foo\n");
  }
};
int MyClass::copyCount = 0;

int main(int argc, const char* argv[]) {
  dispatch_queue_t q = dispatch_queue_create("my.queue", NULL);
  MyClass obj;
  MyClass::printCopyCount();
  void (^block)(void) = ^{
    obj.foo();
  };
  MyClass::printCopyCount();
  dispatch_sync(q, block);
  MyClass::printCopyCount();

  fprintf(stderr, "Done.\n");
  return 0;
}

// CHECK: copyCount = 0
// CHECK: copyCount = 1
// CHECK: MyClass::foo
// CHECK: copyCount = 1
// CHECK: Done.
