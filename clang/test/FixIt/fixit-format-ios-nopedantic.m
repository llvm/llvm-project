// RUN: cp %s %t
// RUN: %clang_cc1 -triple thumbv7-apple-ios8.0.0 -Wformat -Werror -fixit %t

int printf(const char *restrict, ...);
typedef unsigned int NSUInteger;
typedef int NSInteger;
NSUInteger getNSUInteger(void);
NSInteger getNSInteger(void);

void test(void) {
  // For thumbv7-apple-ios8.0.0 the underlying type of ssize_t is long
  // and the underlying type of size_t is unsigned long.

  printf("test 1: %zu", getNSUInteger());

  printf("test 2: %zu %zu", getNSUInteger(), getNSUInteger());

  printf("test 3: %zd", getNSInteger());

  printf("test 4: %zd %zd", getNSInteger(), getNSInteger());
}
