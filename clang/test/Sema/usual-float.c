// RUN: %clang_cc1 %s -fsyntax-only

typedef float CGFloat;

extern void func(CGFloat);
void foo(int dir, int n, int tindex) {
  const float PI = 3.142;
  CGFloat cgf = 3.4;

  float ang = (float) tindex * (-dir*2.0f*PI/n);
  func((CGFloat)cgf/65535.0f);
}
