// RUN: %clang_cc1 -fsyntax-only -Wunused-variable -Wno-unused-const-variable -verify %s

namespace {
  int i = 0; // expected-warning {{unused variable 'i'}}
  const int j = 0;;
}
