// RUN: %clang_cc1 -fsyntax-only -Wlogical-not-parentheses -verify %s
// RUN: not %clang_cc1 -fsyntax-only -Wlogical-not-parentheses -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

bool getBool();
int getInt();

bool test1(int i1, int i2, bool b1, bool b2) {
  bool ret;

  ret = !i1 == i2;
  // expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: warn-logical-not-compare.cpp:[[line:[0-9]*]]:9: warning
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{[[line]]:10-[[line]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:18-[[line]]:18}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{[[line]]:9-[[line]]:9}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:12-[[line]]:12}:")"

  ret = !i1 != i2;
  //expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: warn-logical-not-compare.cpp:[[line:[0-9]*]]:9: warning
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{[[line]]:10-[[line]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:18-[[line]]:18}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{[[line]]:9-[[line]]:9}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:12-[[line]]:12}:")"

  ret = !i1 < i2;
  //expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: warn-logical-not-compare.cpp:[[line:[0-9]*]]:9: warning
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{[[line]]:10-[[line]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:17-[[line]]:17}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{[[line]]:9-[[line]]:9}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:12-[[line]]:12}:")"

  ret = !i1 > i2;
  //expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: warn-logical-not-compare.cpp:[[line:[0-9]*]]:9: warning
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{[[line]]:10-[[line]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:17-[[line]]:17}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{[[line]]:9-[[line]]:9}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:12-[[line]]:12}:")"

  ret = !i1 <= i2;
  //expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: warn-logical-not-compare.cpp:[[line:[0-9]*]]:9: warning
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{[[line]]:10-[[line]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:18-[[line]]:18}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{[[line]]:9-[[line]]:9}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:12-[[line]]:12}:")"

  ret = !i1 >= i2;
  //expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: warn-logical-not-compare.cpp:[[line:[0-9]*]]:9: warning
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{[[line]]:10-[[line]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:18-[[line]]:18}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{[[line]]:9-[[line]]:9}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:12-[[line]]:12}:")"

  ret = i1 == i2;
  ret = i1 != i2;
  ret = i1 < i2;
  ret = i1 > i2;
  ret = i1 <= i2;
  ret = i1 >= i2;

  // Warning silenced by parens.
  ret = (!i1) == i2;
  ret = (!i1) != i2;
  ret = (!i1) < i2;
  ret = (!i1) > i2;
  ret = (!i1) <= i2;
  ret = (!i1) >= i2;

  ret = !b1 == b2;
  ret = !b1 != b2;
  ret = !b1 < b2;
  ret = !b1 > b2;
  ret = !b1 <= b2;
  ret = !b1 >= b2;

  ret = !getInt() == i1;
  // expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: warn-logical-not-compare.cpp:[[line:[0-9]*]]:9: warning
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{[[line]]:10-[[line]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:24-[[line]]:24}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{[[line]]:9-[[line]]:9}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:18-[[line]]:18}:")"

  ret = (!getInt()) == i1;
  ret = !getBool() == b1;
  return ret;
}

enum E {e1, e2};
E getE();

bool test2 (E e) {
  bool ret;
  ret = e == e1;
  ret = e == getE();
  ret = getE() == e1;
  ret = getE() == getE();

  ret = !e == e1;
  // expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: warn-logical-not-compare.cpp:[[line:[0-9]*]]:9: warning
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{[[line]]:10-[[line]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:17-[[line]]:17}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{[[line]]:9-[[line]]:9}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:11-[[line]]:11}:")"

  ret = !e == getE();
  // expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: warn-logical-not-compare.cpp:[[line:[0-9]*]]:9: warning
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{[[line]]:10-[[line]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:21-[[line]]:21}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{[[line]]:9-[[line]]:9}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:11-[[line]]:11}:")"

  ret = !getE() == e1;
  // expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: warn-logical-not-compare.cpp:[[line:[0-9]*]]:9: warning
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{[[line]]:10-[[line]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:22-[[line]]:22}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{[[line]]:9-[[line]]:9}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:16-[[line]]:16}:")"

  ret = !getE() == getE();
  // expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: warn-logical-not-compare.cpp:[[line:[0-9]*]]:9: warning
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{[[line]]:10-[[line]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:26-[[line]]:26}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{[[line]]:9-[[line]]:9}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:16-[[line]]:16}:")"

  ret = !(e == e1);
  ret = !(e == getE());
  ret = !(getE() == e1);
  ret = !(getE() == getE());

  ret = (!e) == e1;
  ret = (!e) == getE();
  ret = (!getE()) == e1;
  ret = (!getE()) == getE();

  return ret;
}

bool test_bitwise_op(int x) {
  bool ret;

  ret = !x & 1;
  // expected-warning@-1 {{logical not is only applied to the left hand side of this bitwise operator}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the bitwise operator first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: warn-logical-not-compare.cpp:[[line:[0-9]*]]:9: warning
  // CHECK: to evaluate the bitwise operator first
  // CHECK: fix-it:"{{.*}}":{[[line]]:10-[[line]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:15-[[line]]:15}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{[[line]]:9-[[line]]:9}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:11-[[line]]:11}:")"
  ret = !(x & 1);
  ret = (!x) & 1;

  // This warning is really about !x & FOO since that's a common misspelling
  // of the negated bit test !(x & FOO).  Don't warn for | and ^, since
  // it's at least conceivable that the user wants to use | as an
  // alternative to || that evaluates both branches.  (The warning above is
  // only emitted if the operand to ! is not a bool, but in C that's common.)
  // And there's no logical ^.
  ret = !x | 1;
  ret = !(x | 1);
  ret = (!x) | 1;

  ret = !x ^ 1;
  ret = !(x ^ 1);
  ret = (!x) ^ 1;

  // These already err, don't also warn.
  !x &= 1; // expected-error{{expression is not assignable}}
  !x |= 1; // expected-error{{expression is not assignable}}
  !x ^= 1; // expected-error{{expression is not assignable}}

  return ret;
}

bool PR16673(int x) {
  bool ret;
  // Make sure we don't emit a fixit for the left paren, but not the right paren.
#define X(x) x 
  ret = X(!x == 1 && 1);
  // expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: warn-logical-not-compare.cpp:[[line:[0-9]*]]:11: warning
  // CHECK: to evaluate the comparison first
  // CHECK-NOT: fix-it
  // CHECK: to silence this warning
  // CHECK-NOT: fix-it
  return ret;
}
