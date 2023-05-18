// RUN: %clang_cc1 %s -emit-llvm -o - | grep 'declare.*printf' | count 1
// PR3612

int printf(const char *, ...);

int foo(void) {
  return printf(printf);
}
