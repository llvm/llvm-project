// RUN: %clang_cc1 -x c++-header -std=c++23 -fallow-pch-with-compiler-errors \
// RUN:   -emit-pch -o %t %S/var-template-partial-spec-from-pch-with-errors.h
// RUN: %clang_cc1 -std=c++23 -fallow-pch-with-compiler-errors -include-pch %t \
// RUN:   -fsyntax-only -verify %s

// expected-no-diagnostics

GH202956::wrapper<int, int> w;
