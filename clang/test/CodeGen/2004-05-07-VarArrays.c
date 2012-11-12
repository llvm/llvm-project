// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

int foo(int len, char arr[][len], int X) {
  return arr[X][0];
}
