// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++17 -verify=expected,both %s
// RUN: %clang_cc1 -std=c++17 -verify=ref,both %s

template<typename T, T val> struct A {};
namespace Temp {
  struct S { int n; };
  constexpr S &addr(S &&s) { return s; }
  A<S &, addr({})> a; // both-error {{reference to temporary object}}
  A<S *, &addr({})> b; // both-error {{pointer to temporary object}}
  A<int &, addr({}).n> c; // both-error {{reference to subobject of temporary object}}
  A<int *, &addr({}).n> d; // both-error {{pointer to subobject of temporary object}}
}
