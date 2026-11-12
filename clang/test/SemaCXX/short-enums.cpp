// RUN: %clang_cc1 -fshort-enums -fsyntax-only %s

// This shouldn't crash: PR9474

enum E { VALUE_1 };

template <typename T>
struct A {};

template <E Enum>
struct B : A<B<Enum> > {};

void bar(int x) {
  switch (x) {
    case sizeof(B<VALUE_1>): ;
  }
}