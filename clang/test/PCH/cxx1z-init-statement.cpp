// Test this without pch.
// RUN: %clang_cc1 -std=c++1z -include %S/cxx1z-init-statement.h -fsyntax-only -emit-llvm -o - %s

// Test with pch.
// RUN: %clang_cc1 -x c++ -std=c++1z -emit-pch -o %t %S/cxx1z-init-statement.h
// RUN: %clang_cc1 -std=c++1z -include-pch %t -fsyntax-only -emit-llvm -o - %s 

void g0(void) {
  static_assert(test_if(-1) == -1, "");
  static_assert(test_if(0) == 0, "");
}

void g1(void) {
  static_assert(test_switch(-1) == -1, "");
  static_assert(test_switch(0) == 0, "");
  static_assert(test_switch(1) == 1, "");
}
