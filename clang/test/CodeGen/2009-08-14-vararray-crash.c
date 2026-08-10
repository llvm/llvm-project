// RUN: %clang_cc1 -emit-llvm < %s

void sum1(int rb) {
  typedef unsigned char imgrow[rb];
  typedef imgrow img[rb];

  const img *br;
  int y;

  (*br)[y];
}
