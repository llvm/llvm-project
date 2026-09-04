// RUN: %clang_cc1 -std=c++14 -verify %s
// RUN: %clang_cc1 -std=c++14 -verify -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -std=c++20 -verify %s
// RUN: %clang_cc1 -std=c++20 -verify -fexperimental-new-constant-interpreter %s
// expected-no-diagnostics

namespace GH203328 {
// Constant evaluation used to crash when it had to find a subobject of an
// object wrapped in _Atomic.

struct Base {
  constexpr void set(int) {}
};

struct Derived : Base {
  // The member call walks to the base-class subobject of 'this'.
  constexpr Derived(int x) { set(x); }
};

struct MemberCall {
  _Atomic(Derived) a;
  constexpr MemberCall(int x) : a(Derived(x)) {}
};

MemberCall mc(0);

struct WithField {
  int x;
  // The assignment walks to the field subobject of 'this'.
  constexpr WithField(int v) : x(v) { x = x + 1; }
};

struct FieldAccess {
  _Atomic(WithField) a;
  constexpr FieldAccess(int v) : a(WithField(v)) {}
};

FieldAccess fa(1);

#if __cplusplus >= 202002L
constinit MemberCall mc2(0);
constinit FieldAccess fa2(1);
#endif
} // namespace GH203328
