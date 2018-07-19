// RUN: cp %s %t
// RUN: %clang_cc1 -triple thumbv7-apple-ios8.0.0 -fsyntax-only -Wformat -fixit %t
// RUN: grep -v CHECK %t | FileCheck %s

int printf(const char * restrict, ...);
typedef unsigned int NSUInteger;
typedef int NSInteger;
NSUInteger getNSUInteger();
NSInteger getNSInteger();

void test() {
  // For thumbv7-apple-ios8.0.0 the underlying type of ssize_t is long
  // and the underlying type of size_t is unsigned long.

  printf("test 1: %zu", getNSUInteger()); 
  // CHECK: printf("test 1: %lu", (unsigned long)getNSUInteger());

  printf("test 2: %zu %zu", getNSUInteger(), getNSUInteger());
  // CHECK: printf("test 2: %lu %lu", (unsigned long)getNSUInteger(), (unsigned long)getNSUInteger());

  printf("test 3: %zd", getNSInteger()); 
  // CHECK: printf("test 3: %ld", (long)getNSInteger());

  printf("test 4: %zd %zd", getNSInteger(), getNSInteger());
  // CHECK: printf("test 4: %ld %ld", (long)getNSInteger(), (long)getNSInteger());
}
