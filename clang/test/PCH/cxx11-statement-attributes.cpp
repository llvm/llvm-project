// Sanity check.
// RUN: %clang_cc1 -include %S/Inputs/cxx11-statement-attributes.h -std=c++11 -Wimplicit-fallthrough -fsyntax-only %s -o - -verify
// Run the same tests, this time with the attributes loaded from the PCH file.
// RUN: %clang_cc1 -x c++-header -emit-pch -std=c++11 -o %t %S/Inputs/cxx11-statement-attributes.h
// RUN: %clang_cc1 -include-pch %t -std=c++11 -Wimplicit-fallthrough -fsyntax-only %s -o - -verify

// Warning from Inputs/cxx11-statement-attributes.h:
// expected-warning@10 {{fallthrough annotation does not directly precede switch label}}

void g(int n) {
  f<1>(n);  // expected-note {{in instantiation of function template specialization 'f<1>' requested here}}
}
