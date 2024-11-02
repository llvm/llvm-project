// PR691
// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
// CHECK: call ptr @llvm.stacksave()

extern void external(int[*]);

void test(int N) {
  int i;
  for (i = 0; i < N; ++i) {
    int VLA[i];
    external(VLA);
  }
}
