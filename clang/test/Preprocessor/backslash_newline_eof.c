// RUN: %clang_cc1 -x c -fsyntax-only -pedantic -verify %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -std=c++98 -pedantic -verify %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -std=c++11 -Wc++98-compat-pedantic -verify=cxx11-compat %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -std=c++11 -verify=cxx11 %s

// cxx11-no-diagnostics

// expected-warning@+4 {{no newline at end of file}}
// expected-note@+3 {{last newline deleted by splice here}}
// cxx11-compat-warning@+2 {{C++98 requires newline at end of file}}
// cxx11-compat-note@+1 {{last newline deleted by splice here}}
int x; \
