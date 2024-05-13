// RUN: %clang_cc1 -o - -emit-llvm -triple x86_64-unknown-linux-gnu -disable-free %s
// We need to use -emit-llvm in order to trigger the error, without it semantic analysis
// does not verify the used bit and there's no error.

char a[1];

void f1(void) {
  int i = 0;
  int j = sizeof(typeof(*(char(*)[i])a));
}
