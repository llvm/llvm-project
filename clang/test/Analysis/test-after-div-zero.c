// RUN: %clang_analyze_cc1 -std=c99 -Dbool=_Bool -analyzer-checker=core,alpha.core.TestAfterDivZero -analyzer-output=text -verify %s
// RUN: %clang_analyze_cc1 -x c++ -analyzer-checker=core,alpha.core.TestAfterDivZero -analyzer-output=text -verify %s

int var;

void err_eq(int x) {
  var = 77 / x; // expected-note {{Division with compared value made here}}
  if (x == 0) { } // expected-warning {{Value being compared against zero has already been used for division}}
} // expected-note@-1 {{Value being compared against zero has already been used for division}}

void err_eq2(int x) {
  var = 77 / x; // expected-note {{Division with compared value made here}}
  if (0 == x) { } // expected-warning {{Value being compared against zero has already been used for division}}
} // expected-note@-1 {{Value being compared against zero has already been used for division}}

void err_ne(int x) {
  var = 77 / x; // expected-note {{Division with compared value made here}}
  if (x != 0) { } // expected-warning {{Value being compared against zero has already been used for division}}
} // expected-note@-1 {{Value being compared against zero has already been used for division}}

void err_ge(int x) {
  var = 77 / x; // expected-note {{Division with compared value made here}}
  if (x >= 0) { } // expected-warning {{Value being compared against zero has already been used for division}}
} // expected-note@-1 {{Value being compared against zero has already been used for division}}

void err_le(int x) {
  var = 77 / x; // expected-note {{Division with compared value made here}}
  if (x <= 0) {} // expected-warning {{Value being compared against zero has already been used for division}}
} // expected-note@-1 {{Value being compared against zero has already been used for division}}

void err_yes(int x) {
  var = 77 / x; // expected-note {{Division with compared value made here}}
  if (x) {} // expected-warning {{Value being compared against zero has already been used for division}}
} // expected-note@-1 {{Value being compared against zero has already been used for division}}
void err_not(int x) {
  var = 77 / x; // expected-note {{Division with compared value made here}}
  if (!x) {} // expected-warning {{Value being compared against zero has already been used for division}}
} // expected-note@-1 {{Value being compared against zero has already been used for division}}

void err_pnot(int x) {
  int *y = &x;
  var = 77 / *y; // expected-note {{Division with compared value made here}}
  if (!x) {} // expected-warning {{Value being compared against zero has already been used for division}}
} // expected-note@-1 {{Value being compared against zero has already been used for division}}

void err_pnot2(int x) {
  int *y = &x;
  var = 77 / x; // expected-note {{Division with compared value made here}}
  if (!*y) {} // expected-warning {{Value being compared against zero has already been used for division}}
} // expected-note@-1 {{Value being compared against zero has already been used for division}}

void err_ppnot(int x) {
  int *y = &x;
  int **z = &y;
  var = 77 / **z; // expected-note {{Division with compared value made here}}
  if (!x) {} // expected-warning {{Value being compared against zero has already been used for division}}
} // expected-note@-1 {{Value being compared against zero has already been used for division}}

void err_orig_checker(int x) {
  if (x != 0) // expected-note {{Assuming 'x' is equal to 0}} expected-note {{Taking false branch}}
    return;
  var = 77 / x; // expected-warning {{Division by zero}} expected-note {{Division by zero}}
  if (!x) {} // no-warning
}

void ok_other(int x, int y) {
  var = 77 / y;
  if (x == 0) {
  }
}

void ok_assign(int x) {
  var = 77 / x;
  x = var / 77; // <- assignment => don't warn
  if (x == 0) {
  }
}

void ok_assign2(int x) {
  var = 77 / x;
  x = var / 77; // <- assignment => don't warn
  if (0 == x) {
  }
}

void ok_dec(int x) {
  var = 77 / x;
  x--; // <- assignment => don't warn
  if (x == 0) {
  }
}

void ok_inc(int x) {
  var = 77 / x;
  x++; // <- assignment => don't warn
  if (x == 0) {
  }
}

void do_something_ptr(int *x);
void ok_callfunc_ptr(int x) {
  var = 77 / x;
  do_something_ptr(&x); // <- pass address of x to function => don't warn
  if (x == 0) {
  }
}

void do_something(int x);
void nok_callfunc(int x) {
  var = 77 / x; // expected-note {{Division with compared value made here}}
  do_something(x);
  if (x == 0) {} // expected-warning {{Value being compared against zero has already been used for division}}
} // expected-note@-1 {{Value being compared against zero has already been used for division}}

void ok_if(int x) {
  if (x > 3)
    var = 77 / x;
  if (x == 0) {
  }
}

void ok_if2(int x) {
  if (x < 3)
    var = 77 / x;
  if (x == 0) {
  } // TODO warn here
}

void ok_pif(int x) {
  int *y = &x;
  if (x < 3)
    var = 77 / *y;
  if (x == 0) {
  } // TODO warn here
}

int getValue(bool *isPositive);
void use(int a);
void foo() {
  bool isPositive;
  int x = getValue(&isPositive);
  if (isPositive) {
    use(5 / x);
  }

  if (x == 0) {
  }
}

int getValue2();
void foo2() {
  int x = getValue2();
  int y = x;

  use(5 / x); // expected-note {{Division with compared value made here}}
  if (y == 0) {} // expected-warning {{Value being compared against zero has already been used for division}}
} // expected-note@-1 {{Value being compared against zero has already been used for division}}

void ok_while(int x) {
  int n = 100 / x;
  while (x != 0) { // <- do not warn
    x--;
  }
}

void err_not2(int x, int y) {
  int v;
  var = 77 / x;

  if (y)
    v = 0;

  if (!x) {
  } // TODO warn here
}

inline void inline_func(int x) {
  var = 77 / x; // expected-note {{Division with compared value made here}}
  if (x == 0) {} // expected-warning {{Value being compared against zero has already been used for division}}
} // expected-note@-1 {{Value being compared against zero has already been used for division}}

void err_inline(int x) {
  var = 77 / x;
  inline_func(x); // expected-note {{Calling 'inline_func'}}
  if (x == 0) {
  }
}

inline void inline_func2(int x) {}

void err_inline2(int x) {
  var = 77 / x; // expected-note {{Division with compared value made here}}
  inline_func2(x);
  if (x == 0) {} // expected-warning {{Value being compared against zero has already been used for division}}
} // expected-note@-1 {{Value being compared against zero has already been used for division}}

inline void inline_func3(int x) {
  var = 77 / x;
}
void ok_inline(int x) {
  var = 77 / x; // expected-note {{Division with compared value made here}}
  inline_func3(x);
  if (x == 0) {} // expected-warning {{Value being compared against zero has already been used for division}}
} // expected-note@-1 {{Value being compared against zero has already been used for division}}
