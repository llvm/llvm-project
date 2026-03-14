// RUN: %clang_cc1 -verify %s -std=c++98 -DEXT
// RUN: %clang_cc1 -verify %s -std=c++98 -Wno-gnu -DNONE
// RUN: %clang_cc1 -verify %s -std=c++98 -Wno-static-float-init -DNONE
// RUN: %clang_cc1 -verify %s -std=c++98 -Wno-gnu-static-float-init -DNONE
// RUN: %clang_cc1 -verify %s -std=c++11 -DERR
// RUN: %clang_cc1 -verify %s -std=c++11 -Wno-gnu -DERR
// RUN: %clang_cc1 -verify %s -std=c++11 -Wno-static-float-init -DNONE
// RUN: %clang_cc1 -verify %s -std=c++11 -Wno-gnu-static-float-init -DERR

#if NONE
// expected-no-diagnostics
#elif ERR
// expected-error@20 {{in-class initializer for static data member of type 'const double' requires 'constexpr' specifier}}
// expected-note@20 {{add 'constexpr'}}
#elif EXT
// expected-warning@20 {{in-class initializer for static data member of type 'const double' is a GNU extension}}
#endif

struct X {
  static const double x = 0.0;
};
