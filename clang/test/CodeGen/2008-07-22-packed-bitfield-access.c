// RUN: %clang_cc1 %s -emit-llvm -o -

int main (void) {
  struct foo {
    unsigned a:16;
    unsigned b:32 __attribute__ ((packed));
  } x;
  x.b = 0x56789abcL;
  return 0;
}
