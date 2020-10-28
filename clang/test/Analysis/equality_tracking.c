// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false

#define NULL (void *)0

#define UCHAR_MAX (unsigned char)(~0U)
#define CHAR_MAX (char)(UCHAR_MAX & (UCHAR_MAX >> 1))
#define CHAR_MIN (char)(UCHAR_MAX & ~(UCHAR_MAX >> 1))

void clang_analyzer_eval(int);
void clang_analyzer_warnIfReached();

int getInt();

void zeroImpliesEquality(int a, int b) {
  clang_analyzer_eval((a - b) == 0); // expected-warning{{UNKNOWN}}
  if ((a - b) == 0) {
    clang_analyzer_eval(b != a);    // expected-warning{{FALSE}}
    clang_analyzer_eval(b == a);    // expected-warning{{TRUE}}
    clang_analyzer_eval(!(a != b)); // expected-warning{{TRUE}}
    clang_analyzer_eval(!(b == a)); // expected-warning{{FALSE}}
    return;
  }
  clang_analyzer_eval((a - b) == 0); // expected-warning{{FALSE}}
  clang_analyzer_eval(b == a);       // expected-warning{{FALSE}}
  clang_analyzer_eval(b != a);       // expected-warning{{TRUE}}
}

void zeroImpliesReversedEqual(int a, int b) {
  clang_analyzer_eval((b - a) == 0); // expected-warning{{UNKNOWN}}
  if ((b - a) == 0) {
    clang_analyzer_eval(b != a); // expected-warning{{FALSE}}
    clang_analyzer_eval(b == a); // expected-warning{{TRUE}}
    return;
  }
  clang_analyzer_eval((b - a) == 0); // expected-warning{{FALSE}}
  clang_analyzer_eval(b == a);       // expected-warning{{FALSE}}
  clang_analyzer_eval(b != a);       // expected-warning{{TRUE}}
}

void canonicalEqual(int a, int b) {
  clang_analyzer_eval(a == b); // expected-warning{{UNKNOWN}}
  if (a == b) {
    clang_analyzer_eval(b == a); // expected-warning{{TRUE}}
    return;
  }
  clang_analyzer_eval(a == b); // expected-warning{{FALSE}}
  clang_analyzer_eval(b == a); // expected-warning{{FALSE}}
}

void test(int a, int b, int c, int d) {
  if (a == b && c == d) {
    if (a == 0 && b == d) {
      clang_analyzer_eval(c == 0); // expected-warning{{TRUE}}
    }
    c = 10;
    if (b == d) {
      clang_analyzer_eval(c == 10); // expected-warning{{TRUE}}
      clang_analyzer_eval(d == 10); // expected-warning{{UNKNOWN}}
                                    // expected-warning@-1{{FALSE}}
      clang_analyzer_eval(b == a);  // expected-warning{{TRUE}}
      clang_analyzer_eval(a == d);  // expected-warning{{TRUE}}

      b = getInt();
      clang_analyzer_eval(a == d); // expected-warning{{TRUE}}
      clang_analyzer_eval(a == b); // expected-warning{{UNKNOWN}}
    }
  }

  if (a != b && b == c) {
    if (c == 42) {
      clang_analyzer_eval(b == 42); // expected-warning{{TRUE}}
      clang_analyzer_eval(a != 42); // expected-warning{{TRUE}}
    }
  }
}

void testIntersection(int a, int b, int c) {
  if (a < 42 && b > 15 && c >= 25 && c <= 30) {
    if (a != b)
      return;

    clang_analyzer_eval(a > 15);  // expected-warning{{TRUE}}
    clang_analyzer_eval(b < 42);  // expected-warning{{TRUE}}
    clang_analyzer_eval(a <= 30); // expected-warning{{UNKNOWN}}

    if (c == b) {
      // For all equal symbols, we should track the minimal common range.
      //
      // Also, it should be noted that c is dead at this point, but the
      // constraint initially associated with c is still around.
      clang_analyzer_eval(a >= 25 && a <= 30); // expected-warning{{TRUE}}
      clang_analyzer_eval(b >= 25 && b <= 30); // expected-warning{{TRUE}}
    }
  }
}

void testPromotion(int a, char b) {
  if (b > 10) {
    if (a == b) {
      // FIXME: support transferring char ranges onto equal int symbols
      //        when char is promoted to int
      clang_analyzer_eval(a > 10);        // expected-warning{{UNKNOWN}}
      clang_analyzer_eval(a <= CHAR_MAX); // expected-warning{{UNKNOWN}}
    }
  }
}

void testPromotionOnlyTypes(int a, char b) {
  if (a == b) {
    // FIXME: support transferring char ranges onto equal int symbols
    //        when char is promoted to int
    clang_analyzer_eval(a <= CHAR_MAX); // expected-warning{{UNKNOWN}}
  }
}

void testDowncast(int a, unsigned char b) {
  if (a <= -10) {
    if ((unsigned char)a == b) {
      // Even though ranges for a and b do not intersect,
      // ranges for (unsigned char)a and b do.
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
    }
    if (a == b) {
      // FIXME: This case on the other hand is different, it shouldn't be
      //        reachable.  However, the corrent symbolic information available
      //        to the solver doesn't allow it to distinguish this expression
      //        from the previous one.
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
    }
  }
}

void testPointers(int *a, int *b, int *c, int *d) {
  if (a == b && c == d) {
    if (a == NULL && b == d) {
      clang_analyzer_eval(c == NULL); // expected-warning{{TRUE}}
    }
  }

  if (a != b && b == c) {
    if (c == NULL) {
      clang_analyzer_eval(a != NULL); // expected-warning{{TRUE}}
    }
  }
}

void testDisequalitiesAfter(int a, int b, int c) {
  if (a >= 10 && b <= 42) {
    if (a == b && c == 15 && c != a) {
      clang_analyzer_eval(b != c);  // expected-warning{{TRUE}}
      clang_analyzer_eval(a != 15); // expected-warning{{TRUE}}
      clang_analyzer_eval(b != 15); // expected-warning{{TRUE}}
      clang_analyzer_eval(b >= 10); // expected-warning{{TRUE}}
      clang_analyzer_eval(a <= 42); // expected-warning{{TRUE}}
    }
  }
}

void testDisequalitiesBefore(int a, int b, int c) {
  if (a >= 10 && b <= 42 && c == 15) {
    if (a == b && c != a) {
      clang_analyzer_eval(b != c);  // expected-warning{{TRUE}}
      clang_analyzer_eval(a != 15); // expected-warning{{TRUE}}
      clang_analyzer_eval(b != 15); // expected-warning{{TRUE}}
      clang_analyzer_eval(b >= 10); // expected-warning{{TRUE}}
      clang_analyzer_eval(a <= 42); // expected-warning{{TRUE}}
    }
  }
}

void avoidInfeasibleConstraintsForClasses(int a, int b) {
  if (a >= 0 && a <= 10 && b >= 20 && b <= 50) {
    if ((b - a) == 0) {
      clang_analyzer_warnIfReached(); // no warning
    }
    if (a == b) {
      clang_analyzer_warnIfReached(); // no warning
    }
    if (a != b) {
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
    } else {
      clang_analyzer_warnIfReached(); // no warning
    }
  }
}

void avoidInfeasibleConstraintforGT(int a, int b) {
  int c = b - a;
  if (c <= 0)
    return;
  // c > 0
  // b - a > 0
  // b > a
  if (a != b) {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
    return;
  }
  clang_analyzer_warnIfReached(); // no warning
  // a == b
  if (c < 0)
    ;
}

void avoidInfeasibleConstraintforLT(int a, int b) {
  int c = b - a;
  if (c >= 0)
    return;
  // c < 0
  // b - a < 0
  // b < a
  if (a != b) {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
    return;
  }
  clang_analyzer_warnIfReached(); // no warning
  // a == b
  if (c < 0)
    ;
}
