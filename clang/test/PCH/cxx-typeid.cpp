// Test this without pch.
// RUN: %clang_cc1 -include %S/cxx-typeid.h -fsyntax-only -stdlib=libstdc++ -verify %s

// RUN: %clang_cc1 -x c++-header -emit-pch -stdlib=libstdc++ -o %t.pch %S/cxx-typeid.h
// RUN: %clang_cc1 -include-pch %t.pch -fsyntax-only -stdlib=libstdc++ -verify %s

// expected-no-diagnostics

void f() {
    (void)typeid(int);
}
