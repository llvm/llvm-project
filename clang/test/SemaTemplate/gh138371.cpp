// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// This would previously trigger a failed assertion when instantiating the
// template which uses an overloaded call operator because the end location
// for the expression came from a macro expansion.

#define ASSIGN_OR_RETURN(...)  (__VA_ARGS__)

struct Loc {
  int operator()(const char* _Nonnull f = __builtin_FILE()) const;
};

template <typename Ty>
void f() {
  ASSIGN_OR_RETURN(Loc()());
}

void test() {
  f<int>();
}

