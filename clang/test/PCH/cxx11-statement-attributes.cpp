// Basic correctness check.
// RUN: %clang_cc1 -include %S/Inputs/cxx11-statement-attributes.h -std=c++11 -Wimplicit-fallthrough -fsyntax-only %s -o - -verify
// RUN: %clang_cc1 -include %S/Inputs/cxx11-statement-attributes.h -std=c++1z -Wimplicit-fallthrough -fsyntax-only %s -o - -verify
// Run the same tests, this time with the attributes loaded from the PCH file.
// RUN: %clang_cc1 -x c++-header -emit-pch -std=c++11 -o %t %S/Inputs/cxx11-statement-attributes.h
// RUN: %clang_cc1 -include-pch %t -std=c++11 -Wimplicit-fallthrough -fsyntax-only %s -o - -verify
// RUN: %clang_cc1 -x c++-header -emit-pch -std=c++1z -o %t %S/Inputs/cxx11-statement-attributes.h
// RUN: %clang_cc1 -include-pch %t -std=c++1z -Wimplicit-fallthrough -fsyntax-only %s -o - -verify

// expected-warning@Inputs/cxx11-statement-attributes.h:10 {{unannotated fall-through}}
// expected-note-re@Inputs/cxx11-statement-attributes.h:10 {{insert '[[{{(clang::)?}}fallthrough]];'}}
// expected-note@Inputs/cxx11-statement-attributes.h:10 {{insert 'break;'}}

void g(int n) {
  f<1>(n);  // expected-note {{in instantiation of function template specialization 'f<1>' requested here}}
}
