// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s -verify=expected,not-cxx20
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -Wdeprecated -verify %s -verify=expected,cxx20
// RUN: %clang_cc1 -std=c++26 -fsyntax-only -Wdeprecated -verify %s -verify=expected,cxx26

void f(int (&array1)[2], int (&array2)[2]) {
  if (array1 == array2) { } // not-cxx20-warning {{comparison between two arrays compare their addresses}} cxx20-warning {{comparison between two arrays is deprecated}}
                            // cxx26-error@-1 {{comparison between two arrays is ill-formed in C++26}}
}
