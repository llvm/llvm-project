// RUN: %clang_cc1 %s -emit-llvm     -o -
// RUN: %clang_cc1 %s -emit-llvm -O1 -o -

static int bar();
void foo() {
  int a = bar();
}
int bar(unsigned a) {
}
