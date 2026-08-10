// RUN: %clang_cc1 -fsyntax-only -Wlogical-not-parentheses -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wlogical-not-parentheses -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

int getInt(void);

int test1(int i1, int i2) {
  int ret;

  ret = !i1 == i2;
  // expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: warn-logical-not-compare.c:[[line:[0-9]*]]:9: warning
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
  // CHECK: warn-logical-not-compare.c:[[line:[0-9]*]]:9: warning
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
  // CHECK: warn-logical-not-compare.c:[[line:[0-9]*]]:9: warning
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
  // CHECK: warn-logical-not-compare.c:[[line:[0-9]*]]:9: warning
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
  // CHECK: warn-logical-not-compare.c:[[line:[0-9]*]]:9: warning
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
  // CHECK: warn-logical-not-compare.c:[[line:[0-9]*]]:9: warning
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

  ret = !getInt() == i1;
  // expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: warn-logical-not-compare.c:[[line:[0-9]*]]:9: warning
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{[[line]]:10-[[line]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:24-[[line]]:24}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{[[line]]:9-[[line]]:9}:"("
  // CHECK: fix-it:"{{.*}}":{[[line]]:18-[[line]]:18}:")"

  ret = (!getInt()) == i1;
  return ret;
}

enum E {e1, e2};
enum E getE(void);

int test2 (enum E e) {
  int ret;
  ret = e == e1;
  ret = e == getE();
  ret = getE() == e1;
  ret = getE() == getE();

  ret = !e == e1;
  // expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: warn-logical-not-compare.c:[[line:[0-9]*]]:9: warning
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
  // CHECK: warn-logical-not-compare.c:[[line:[0-9]*]]:9: warning
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
  // CHECK: warn-logical-not-compare.c:[[line:[0-9]*]]:9: warning
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
  // CHECK: warn-logical-not-compare.c:[[line:[0-9]*]]:9: warning
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

int PR16673(int x) {
  int ret;
  // Make sure we don't emit a fixit for the left paren, but not the right paren.
#define X(x) x
  ret = X(!x == 1 && 1);
  // expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: warn-logical-not-compare.c:[[line:[0-9]*]]:11: warning
  // CHECK: to evaluate the comparison first
  // CHECK-NOT: fix-it
  // CHECK: to silence this warning
  // CHECK-NOT: fix-it
  return ret;
}

int compare_pointers(int* a, int* b) {
  int ret;
  ret = !!a == !!b;
  ret = !!a != !!b;
  return ret;
}
