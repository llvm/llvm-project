// RUN: %clang_cc1  %s -emit-llvm -o - | grep getelementptr

int *test(int *X, int Y) {
  return X + Y;
}
