// RUN: %clang_cc1 -fsyntax-only -verify -fno-rtti %s

namespace std {
  class type_info;
}

void f()
{
  (void)typeid(int); // expected-error {{cannot use typeid with -fno-rtti}}
}
