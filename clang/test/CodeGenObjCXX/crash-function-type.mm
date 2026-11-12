// Mark test as unsupported on PS5 due to PS5 doesn't support function sanitizer.
// UNSUPPORTED: target=x86_64-sie-ps5

// RUN: %clang_cc1 -fblocks -fsanitize=function -emit-llvm %s -o %t

void g(void (^)());
void f() {
  __block int a = 0;
  g(^() {
    a++;
  });
}
