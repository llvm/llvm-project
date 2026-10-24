// RUN: %clang_cc1 -triple x86_64-win32 -fsyntax-only -fms-extensions \
// RUN:   -std=c++20 -verify %s

struct Outer {
  template <typename T>
  struct Nested {
    __declspec(dllexport) Nested(int = 0) {} // expected-error{{'__declspec(dllexport)' cannot be applied to more than one default constructor}}
    __declspec(dllexport) Nested(double = 0) {} // expected-note{{declared here}}
  };
};

template struct Outer::Nested<int>; // expected-note{{in instantiation of template class 'Outer::Nested<int>' requested here}}

struct PackOuter {
  template <typename... T>
  struct Nested {
    __declspec(dllexport) Nested(T..., int = 0) {} // expected-error{{'__declspec(dllexport)' cannot be applied to more than one default constructor}}
    __declspec(dllexport) Nested(double = 0) {} // expected-note{{declared here}}
  };
};

// The first constructor is not a default constructor after T expands to int.
template struct PackOuter::Nested<int>;

// With an empty pack, both constructors can be called without arguments.
template struct PackOuter::Nested<>; // expected-note{{in instantiation of template class 'PackOuter::Nested<>' requested here}}

template <typename T>
concept Small = sizeof(T) == 1;

struct ConstraintOuter {
  template <typename T>
  struct Nested {
    __declspec(dllexport) Nested() requires Small<T> {} // expected-error{{'__declspec(dllexport)' cannot be applied to more than one default constructor}}
    __declspec(dllexport) Nested(int = 0) {} // expected-note{{declared here}}
  };
};

// The constrained constructor is not viable here.
template struct ConstraintOuter::Nested<int>;

// Both constructors are viable here.
template struct ConstraintOuter::Nested<char>; // expected-note{{in instantiation of template class 'ConstraintOuter::Nested<char>' requested here}}
