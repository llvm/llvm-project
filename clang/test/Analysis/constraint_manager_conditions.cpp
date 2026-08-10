// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

void comparison_lt(int x, int y) {
  if (x < y) {
    clang_analyzer_eval(x < y);  // expected-warning{{TRUE}}
    clang_analyzer_eval(y > x);  // expected-warning{{TRUE}}
    clang_analyzer_eval(x > y);  // expected-warning{{FALSE}}
    clang_analyzer_eval(y < x);  // expected-warning{{FALSE}}
    clang_analyzer_eval(x <= y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y >= x); // expected-warning{{TRUE}}
    clang_analyzer_eval(x >= y); // expected-warning{{FALSE}}
    clang_analyzer_eval(y <= x); // expected-warning{{FALSE}}
    clang_analyzer_eval(x == y); // expected-warning{{FALSE}}
    clang_analyzer_eval(y == x); // expected-warning{{FALSE}}
    clang_analyzer_eval(x != y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y != x); // expected-warning{{TRUE}}
  } else {
    clang_analyzer_eval(x < y);  // expected-warning{{FALSE}}
    clang_analyzer_eval(y > x);  // expected-warning{{FALSE}}
    clang_analyzer_eval(x > y);  // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y < x);  // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x <= y); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y >= x); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x >= y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y <= x); // expected-warning{{TRUE}}
    clang_analyzer_eval(x == y); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y == x); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x != y); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y != x); // expected-warning{{TRUE}} expected-warning{{FALSE}}
  }
}

void comparison_gt(int x, int y) {
  if (x > y) {
    clang_analyzer_eval(x < y);  // expected-warning{{FALSE}}
    clang_analyzer_eval(y > x);  // expected-warning{{FALSE}}
    clang_analyzer_eval(x > y);  // expected-warning{{TRUE}}
    clang_analyzer_eval(y < x);  // expected-warning{{TRUE}}
    clang_analyzer_eval(x <= y); // expected-warning{{FALSE}}
    clang_analyzer_eval(y >= x); // expected-warning{{FALSE}}
    clang_analyzer_eval(x >= y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y <= x); // expected-warning{{TRUE}}
    clang_analyzer_eval(x == y); // expected-warning{{FALSE}}
    clang_analyzer_eval(y == x); // expected-warning{{FALSE}}
    clang_analyzer_eval(x != y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y != x); // expected-warning{{TRUE}}
  } else {
    clang_analyzer_eval(x < y);  // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y > x);  // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x > y);  // expected-warning{{FALSE}}
    clang_analyzer_eval(y < x);  // expected-warning{{FALSE}}
    clang_analyzer_eval(x <= y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y >= x); // expected-warning{{TRUE}}
    clang_analyzer_eval(x >= y); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y <= x); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x == y); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y == x); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x != y); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y != x); // expected-warning{{TRUE}} expected-warning{{FALSE}}
  }
}

void comparison_le(int x, int y) {
  if (x <= y) {
    clang_analyzer_eval(x < y);  // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y > x);  // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x > y);  // expected-warning{{FALSE}}
    clang_analyzer_eval(y < x);  // expected-warning{{FALSE}}
    clang_analyzer_eval(x <= y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y >= x); // expected-warning{{TRUE}}
    clang_analyzer_eval(x >= y); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y <= x); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x == y); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y == x); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x != y); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y != x); // expected-warning{{TRUE}} expected-warning{{FALSE}}
  } else {
    clang_analyzer_eval(x < y);  // expected-warning{{FALSE}}
    clang_analyzer_eval(y > x);  // expected-warning{{FALSE}}
    clang_analyzer_eval(x > y);  // expected-warning{{TRUE}}
    clang_analyzer_eval(y < x);  // expected-warning{{TRUE}}
    clang_analyzer_eval(x <= y); // expected-warning{{FALSE}}
    clang_analyzer_eval(y >= x); // expected-warning{{FALSE}}
    clang_analyzer_eval(x >= y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y <= x); // expected-warning{{TRUE}}
    clang_analyzer_eval(x == y); // expected-warning{{FALSE}}
    clang_analyzer_eval(y == x); // expected-warning{{FALSE}}
    clang_analyzer_eval(x != y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y != x); // expected-warning{{TRUE}}
  }
}

void comparison_ge(int x, int y) {
  if (x >= y) {
    clang_analyzer_eval(x < y);  // expected-warning{{FALSE}}
    clang_analyzer_eval(y > x);  // expected-warning{{FALSE}}
    clang_analyzer_eval(x > y);  // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y < x);  // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x <= y); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y >= x); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x >= y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y <= x); // expected-warning{{TRUE}}
    clang_analyzer_eval(x == y); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y == x); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x != y); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y != x); // expected-warning{{TRUE}} expected-warning{{FALSE}}
  } else {
    clang_analyzer_eval(x < y);  // expected-warning{{TRUE}}
    clang_analyzer_eval(y > x);  // expected-warning{{TRUE}}
    clang_analyzer_eval(x > y);  // expected-warning{{FALSE}}
    clang_analyzer_eval(y < x);  // expected-warning{{FALSE}}
    clang_analyzer_eval(x <= y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y >= x); // expected-warning{{TRUE}}
    clang_analyzer_eval(x >= y); // expected-warning{{FALSE}}
    clang_analyzer_eval(y <= x); // expected-warning{{FALSE}}
    clang_analyzer_eval(x == y); // expected-warning{{FALSE}}
    clang_analyzer_eval(y == x); // expected-warning{{FALSE}}
    clang_analyzer_eval(x != y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y != x); // expected-warning{{TRUE}}
  }
}

void comparison_eq(int x, int y) {
  if (x == y) {
    clang_analyzer_eval(x < y);  // expected-warning{{FALSE}}
    clang_analyzer_eval(y > x);  // expected-warning{{FALSE}}
    clang_analyzer_eval(x > y);  // expected-warning{{FALSE}}
    clang_analyzer_eval(y < x);  // expected-warning{{FALSE}}
    clang_analyzer_eval(x <= y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y >= x); // expected-warning{{TRUE}}
    clang_analyzer_eval(x >= y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y <= x); // expected-warning{{TRUE}}
    clang_analyzer_eval(x == y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y == x); // expected-warning{{TRUE}}
    clang_analyzer_eval(x != y); // expected-warning{{FALSE}}
    clang_analyzer_eval(y != x); // expected-warning{{FALSE}}
  } else {
    clang_analyzer_eval(x < y);  // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y > x);  // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x > y);  // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y < x);  // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x <= y); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y >= x); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x >= y); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y <= x); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x == y); // expected-warning{{FALSE}}
    clang_analyzer_eval(y == x); // expected-warning{{FALSE}}
    clang_analyzer_eval(x != y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y != x); // expected-warning{{TRUE}}
  }
}

void comparison_ne(int x, int y) {
  if (x != y) {
    clang_analyzer_eval(x < y);  // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y > x);  // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x > y);  // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y < x);  // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x <= y); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y >= x); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x >= y); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(y <= x); // expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(x == y); // expected-warning{{FALSE}}
    clang_analyzer_eval(y == x); // expected-warning{{FALSE}}
    clang_analyzer_eval(x != y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y != x); // expected-warning{{TRUE}}
  } else {
    clang_analyzer_eval(x < y);  // expected-warning{{FALSE}}
    clang_analyzer_eval(y > x);  // expected-warning{{FALSE}}
    clang_analyzer_eval(x > y);  // expected-warning{{FALSE}}
    clang_analyzer_eval(y < x);  // expected-warning{{FALSE}}
    clang_analyzer_eval(x <= y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y >= x); // expected-warning{{TRUE}}
    clang_analyzer_eval(x >= y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y <= x); // expected-warning{{TRUE}}
    clang_analyzer_eval(x == y); // expected-warning{{TRUE}}
    clang_analyzer_eval(y == x); // expected-warning{{TRUE}}
    clang_analyzer_eval(x != y); // expected-warning{{FALSE}}
    clang_analyzer_eval(y != x); // expected-warning{{FALSE}}
  }
}

void comparison_le_ne(int x, int y) {
  if (x <= y)
    if (x != y) {
      clang_analyzer_eval(x < y);  // expected-warning{{TRUE}}
      clang_analyzer_eval(y > x);  // expected-warning{{TRUE}}
      clang_analyzer_eval(x >= y); // expected-warning{{FALSE}}
      clang_analyzer_eval(y <= x); // expected-warning{{FALSE}}
    }
}

void comparison_ge_ne(int x, int y) {
  if (x >= y)
    if (x != y) {
      clang_analyzer_eval(x > y);  // expected-warning{{TRUE}}
      clang_analyzer_eval(y < x);  // expected-warning{{TRUE}}
      clang_analyzer_eval(x <= y); // expected-warning{{FALSE}}
      clang_analyzer_eval(y >= x); // expected-warning{{FALSE}}
    }
}

void comparison_le_ge(int x, int y) {
  if (x <= y)
    if (x >= y) {
      clang_analyzer_eval(x == y); // expected-warning{{TRUE}}
      clang_analyzer_eval(y == x); // expected-warning{{TRUE}}
      clang_analyzer_eval(x != y); // expected-warning{{FALSE}}
      clang_analyzer_eval(y != x); // expected-warning{{FALSE}}
    }
}

// Test the logic of reaching the `Unknonw` tristate in CmpOpTable.
void cmp_op_table_unknownX2(int x, int y, int z) {
  if (x >= y) {
                    // x >= y    [1, 1]
    if (x + z < y)
      return;
                    // x + z < y [0, 0]
    if (z != 0)
      return;
                    // x < y     [0, 0]
    clang_analyzer_eval(x > y);  // expected-warning{{TRUE}} expected-warning{{FALSE}}
  }
}
