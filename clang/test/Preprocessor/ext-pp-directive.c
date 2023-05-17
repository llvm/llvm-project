// For C
// RUN: %clang_cc1 -std=c99 -fsyntax-only -verify=pre-c2x-pedantic -pedantic %s
// RUN: %clang_cc1 -std=c2x -fsyntax-only -verify=pre-c2x-compat -Wpre-c2x-compat %s
// RUN: not %clang_cc1 -std=c99 -fsyntax-only -verify %s
// RUN: not %clang_cc1 -std=c2x -fsyntax-only -verify -pedantic %s
// RUN: not %clang_cc1 -std=c2x -fsyntax-only -verify %s

// For C++
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify=pre-cpp23-pedantic -pedantic %s
// RUN: %clang_cc1 -x c++ -std=c++23 -fsyntax-only -verify=pre-cpp23-compat -Wpre-c++23-compat %s
// RUN: not %clang_cc1 -x c++ -fsyntax-only -verify %s
// RUN: not %clang_cc1 -x c++ -std=c++23 -fsyntax-only -verify -pedantic %s
// RUN: not %clang_cc1 -x c++ -std=c++23 -fsyntax-only -verify %s

int x;

#if 1
#elifdef A // #1
#endif
// For C
// pre-c2x-pedantic-warning@#1 {{use of a '#elifdef' directive is a C2x extension}}
// pre-c2x-compat-warning@#1 {{use of a '#elifdef' directive is incompatible with C standards before C2x}}

// For C++
// pre-cpp23-pedantic-warning@#1 {{use of a '#elifdef' directive is a C++23 extension}}
// pre-cpp23-compat-warning@#1 {{use of a '#elifdef' directive is incompatible with C++ standards before C++23}}

#if 1
#elifndef B // #2
#endif
// For C
// pre-c2x-pedantic-warning@#2 {{use of a '#elifndef' directive is a C2x extension}}
// pre-c2x-compat-warning@#2 {{use of a '#elifndef' directive is incompatible with C standards before C2x}}

// For C++
// pre-cpp23-pedantic-warning@#2 {{use of a '#elifndef' directive is a C++23 extension}}
// pre-cpp23-compat-warning@#2 {{use of a '#elifndef' directive is incompatible with C++ standards before C++23}}

#if 0
#elifdef C
#endif
// For C
// pre-c2x-pedantic-warning@-3 {{use of a '#elifdef' directive is a C2x extension}}
// pre-c2x-compat-warning@-4 {{use of a '#elifdef' directive is incompatible with C standards before C2x}}

// For C++
// pre-cpp23-pedantic-warning@-7 {{use of a '#elifdef' directive is a C++23 extension}}
// pre-cpp23-compat-warning@-8 {{use of a '#elifdef' directive is incompatible with C++ standards before C++23}}

#if 0
#elifndef D
#endif
// For C
// pre-c2x-pedantic-warning@-3 {{use of a '#elifndef' directive is a C2x extension}}
// pre-c2x-compat-warning@-4 {{use of a '#elifndef' directive is incompatible with C standards before C2x}}

// For C++
// pre-cpp23-pedantic-warning@-7 {{use of a '#elifndef' directive is a C++23 extension}}
// pre-cpp23-compat-warning@-8 {{use of a '#elifndef' directive is incompatible with C++ standards before C++23}}

#warning foo
// For C
// pre-c2x-pedantic-warning@-2 {{#warning is a C2x extension}}
// pre-c2x-pedantic-warning@-3 {{foo}}
// pre-c2x-compat-warning@-4 {{#warning is incompatible with C standards before C2x}}
// pre-c2x-compat-warning@-5 {{foo}}

// For C++
// pre-cpp23-pedantic-warning@-8 {{#warning is a C++23 extension}}
// pre-cpp23-pedantic-warning@-9 {{foo}}
// pre-cpp23-compat-warning@-10 {{#warning is incompatible with C++ standards before C++23}}
// pre-cpp23-compat-warning@-11 {{foo}}
