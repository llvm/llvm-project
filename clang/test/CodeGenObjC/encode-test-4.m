// RUN: %clang_cc1 -emit-llvm -o - %s -O2 | grep "ret i32 1"
typedef long Integer;
typedef enum : Integer { Red, Green, Blue} Color;
typedef enum { Cyan, Magenta, Yellow, Key } PrintColor;

int a() {
  return @encode(int) == @encode(int) &&
    @encode(Color) == @encode(long) &&
    @encode(PrintColor) == @encode(int);
}
