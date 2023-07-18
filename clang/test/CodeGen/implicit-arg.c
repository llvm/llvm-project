// RUN: %clang_cc1 %s -emit-llvm     -o -
// RUN: %clang_cc1 %s -emit-llvm -O1 -o -
// rdar://6518089

static int bar();
void foo() {
  int a = bar();
}
int bar(unsigned a) {
}
