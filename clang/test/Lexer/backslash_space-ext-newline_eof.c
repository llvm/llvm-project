// RUN: %clang_cc1 -x c -fsyntax-only -pedantic -verify=expected,no-newline %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -std=c++98 -pedantic -verify=expected,no-newline %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -std=c++11 -Wc++98-compat-pedantic -verify=expected,no-newline-compat %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -std=c++11 -verify=expected,cxx11 %s

// no-newline-warning@+6 {{no newline at end of file}}
// no-newline-note@+5 {{last newline deleted by splice here}}
// no-newline-compat-warning@+4 {{C++98 requires newline at end of file}}
// no-newline-compat-note@+3 {{last newline deleted by splice here}}
// expected-warning@+2 {{backslash and newline separated by space}}
// The next line intentionally has a trailing tab character.
int x; \	
