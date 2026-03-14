// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null



union foo { int X; };

void test(union foo* F) {
  {
    union foo { float X; } A;
  }
}
