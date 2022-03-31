// RUN: %clang_cc1 -emit-llvm -o - -triple x86_64-apple-darwin10 -fblocks %s
// RUN: %clang_cc1 -emit-llvm -o - -triple i386-apple-darwin10 -fblocks %s
typedef int __attribute__((aligned(32)))  ai;

void f(void) {
  __block ai a = 10;

  ^{
    a = 20;
  }();
}

void g(void) {
  __block double a = 10;

  ^{
    a = 20;
  }();
}
