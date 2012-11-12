// PR691
// RUN: %clang_cc1 %s -emit-llvm -o - | opt -std-compile-opts | \
// RUN:    llvm-dis | grep llvm.stacksave

void test(int N) {
  int i;
  for (i = 0; i < N; ++i) {
    int VLA[i];
    external(VLA);
  }
}
