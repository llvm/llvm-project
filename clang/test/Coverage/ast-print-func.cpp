// RUN: %clang_cc1 -ast-print -std=c++14 %s -o %t.1.cpp
// RUN: %clang_cc1 -ast-print -std=c++14 %t.1.cpp -o %t.2.cpp
// RUN: diff %t.1.cpp %t.2.cpp

auto func_01(int, char) -> double;

auto func_02(int x) -> int { return 2 + x; }

void func_03() {
  extern void g(), h();
  return;
}
