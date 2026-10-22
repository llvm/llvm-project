// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.cstring -verify %s

// Test that a nonstandard 'memset' does not trigger an assertion failure.
// Code adapted from https://github.com/llvm/llvm-project/issues/228865

// expected-no-diagnostics

struct B {
} b;

struct A {
  struct B b;
} a;

// Not the standard 'memset' -- there the third parameter would be a 'size_t'.
void *memset(void *s, int c, const unsigned &cond);

void foo() {
  memset(&a.b, 0, sizeof(b)); // no-crash
}
