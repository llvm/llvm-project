// RUN: %clang_cc1 -fsyntax-only -verify -Wtautological-negation-compare -Wno-constant-logical-operand %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wtautological-compare -Wno-constant-logical-operand %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wall -Wno-unused -Wno-loop-analysis -Wno-constant-logical-operand %s

#define COPY(x) x

void test_int(int x) {
  if (x || !x) {} // expected-warning {{'||' of a value and its negation always evaluates to true}}
  if (!x || x) {} // expected-warning {{'||' of a value and its negation always evaluates to true}}
  if (x && !x) {} // expected-warning {{'&&' of a value and its negation always evaluates to false}}
  if (!x && x) {} // expected-warning {{'&&' of a value and its negation always evaluates to false}}

  // parentheses are ignored
  if (x || (!x)) {} // expected-warning {{'||' of a value and its negation always evaluates to true}}
  if (!(x) || x) {} // expected-warning {{'||' of a value and its negation always evaluates to true}}

  // don't warn on macros
  if (COPY(x) || !x) {}
  if (!x || COPY(x)) {}
  if (x && COPY(!x)) {}
  if (COPY(!x && x)) {}

  // dont' warn on literals
  if (1 || !1) {}
  if (!42 && 42) {}


  // don't warn on overloads
  struct Foo{
    int val;
    Foo operator!() const { return Foo{!val}; }
    bool operator||(const Foo other) const { return val || other.val; }
    bool operator&&(const Foo other) const { return val && other.val; }
  };

  Foo f{3};
  if (f || !f) {}
  if (!f || f) {}
  if (f.val || !f.val) {} // expected-warning {{'||' of a value and its negation always evaluates to true}}
  if (!f.val && f.val) {} // expected-warning {{'&&' of a value and its negation always evaluates to false}}
}
