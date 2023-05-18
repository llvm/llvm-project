// RUN: %clang_cc1 -x c %s -emit-llvm -o /dev/null

int test(_Bool pos, _Bool color) {
  return 0;
  return (pos && color);
}
