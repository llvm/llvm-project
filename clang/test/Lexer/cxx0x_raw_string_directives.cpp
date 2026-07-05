// RUN: %clang_cc1 -std=c++11 -fsyntax-only %s
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -verify %s

// expected-error@8 {{in c++98 only}}
#if 0
R"(
#else
#error in c++98 only)"
#endif
