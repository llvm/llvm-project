// RUN: %clang_cc1 -std=c++2c -verify=compat -fsyntax-only -Wpre-c++26-compat %s
// RUN: %clang_cc1 -std=c++11 -verify=pre2c -fsyntax-only -Wc++26-extensions %s

struct S {
  friend int, long, char; // compat-warning {{variadic 'friend' declarations are incompatible with C++ standards before C++2c}} \
                          // pre2c-warning {{variadic 'friend' declarations are a C++2c extension}}
};

template <typename ...Types>
struct TS {
  friend Types...; // compat-warning {{variadic 'friend' declarations are incompatible with C++ standards before C++2c}} \
                   // pre2c-warning {{variadic 'friend' declarations are a C++2c extension}}

  friend int, Types..., Types...; // compat-warning {{variadic 'friend' declarations are incompatible with C++ standards before C++2c}} \
                                  // pre2c-warning {{variadic 'friend' declarations are a C++2c extension}}
};
