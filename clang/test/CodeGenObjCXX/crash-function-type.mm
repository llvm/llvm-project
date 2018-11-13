// RUN: %clang_cc1 -fblocks -fsanitize=function -emit-llvm %s -o %t

void g(void (^)());
void f() {
  __block int a = 0;
  g(^() {
    a++;
  });
}
