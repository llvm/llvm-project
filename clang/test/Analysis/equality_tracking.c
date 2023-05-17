// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false

#define NULL (void *)0

#define UCHAR_MAX (unsigned char)(~0U)
#define CHAR_MAX (char)(UCHAR_MAX & (UCHAR_MAX >> 1))
#define CHAR_MIN (char)(UCHAR_MAX & ~(UCHAR_MAX >> 1))

void clang_analyzer_value(int);
void clang_analyzer_eval(int);
void clang_analyzer_warnIfReached(void);

int getInt(void);

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

void implyDisequalityFromGT(int a, int b) {
  if (a > b) {
    clang_analyzer_eval(a == b); // expected-warning{{FALSE}}
    clang_analyzer_eval(a != b); // expected-warning{{TRUE}}
  }
}

void implyDisequalityFromLT(int a, int b) {
  if (a < b) {
    clang_analyzer_eval(a == b); // expected-warning{{FALSE}}
    clang_analyzer_eval(a != b); // expected-warning{{TRUE}}
  }
}

void deletePointBefore(int x, int tmp) {
  if(tmp == 0)
    if(x != tmp)
     clang_analyzer_value(x); // expected-warning {{32s:{ [-2147483648, -1], [1, 2147483647] }}}
}

void deletePointAfter(int x, int tmp) {
  if(x != tmp)
    if(tmp == 2147483647)
      clang_analyzer_value(x); // expected-warning {{32s:{ [-2147483648, 2147483646] }}}
}

void deleteTwoPoints(int x, int tmp1, int tmp2) {
  if(x != tmp1) {
    if (tmp1 == 42 && tmp2 == 87) {
      clang_analyzer_value(x); // expected-warning {{32s:{ [-2147483648, 41], [43, 2147483647] }}}
      if(x != tmp2)
        clang_analyzer_value(x); // expected-warning {{32s:{ [-2147483648, 41], [43, 86], [88, 2147483647] }}}
    }
  }
}

void deleteAllPoints(unsigned char x, unsigned char *arr) {

#define cond(n) \
arr[n##0] == n##0 && \
arr[n##1] == n##1 && \
arr[n##2] == n##2 && \
arr[n##3] == n##3 && \
arr[n##4] == n##4 && \
arr[n##5] == n##5 && \
arr[n##6] == n##6 && \
arr[n##7] == n##7 && \
arr[n##8] == n##8 && \
arr[n##9] == n##9 && \

#define condX(n) \
arr[n##0] != x && \
arr[n##1] != x && \
arr[n##2] != x && \
arr[n##3] != x && \
arr[n##4] != x && \
arr[n##5] != x && \
arr[n##6] != x && \
arr[n##7] != x && \
arr[n##8] != x && \
arr[n##9] != x && \

  clang_analyzer_value(x); // expected-warning {{{ [0, 255] }}}
  if (
    cond()  // 0  .. 9
    cond(1) // 10 .. 19
    cond(2) // 20 .. 29
    cond(3) // 30 .. 39
    cond(4) // 40 .. 49
    cond(5) // 50 .. 59
    cond(6) // 60 .. 69
    cond(7) // 70 .. 79
    cond(8) // 80 .. 89
    cond(9) // 90 .. 99
    cond(10) // 100 .. 209
    cond(11) // 110 .. 219
    cond(12) // 120 .. 229
    cond(13) // 130 .. 239
    cond(14) // 140 .. 249
    cond(15) // 150 .. 259
    cond(16) // 160 .. 269
    cond(17) // 170 .. 279
    cond(18) // 180 .. 289
    cond(19) // 190 .. 199
    cond(20) // 200 .. 209
    cond(21) // 210 .. 219
    cond(22) // 220 .. 229
    cond(23) // 230 .. 239
    cond(24) // 240 .. 249
    arr[250] == 250 &&
    arr[251] == 251 &&
    arr[252] == 252 &&
    arr[253] == 253 &&
    arr[254] == 254 &&
    arr[255] == 255
    ) {
    if (
      condX()  // 0  .. 9
      condX(1) // 10 .. 19
      condX(2) // 20 .. 29
      condX(3) // 30 .. 39
      condX(4) // 40 .. 49
      condX(5) // 50 .. 59
      condX(6) // 60 .. 69
      condX(7) // 70 .. 79
      condX(8) // 80 .. 89
      condX(9) // 90 .. 99
      condX(10) // 100 .. 209
      condX(11) // 110 .. 219
      condX(12) // 120 .. 229
      condX(13) // 130 .. 239
      condX(14) // 140 .. 249
      condX(15) // 150 .. 259
      condX(16) // 160 .. 269
      condX(17) // 170 .. 279
      condX(18) // 180 .. 289
      condX(19) // 190 .. 199
      condX(20) // 200 .. 209
      condX(21) // 210 .. 219
      condX(22) // 220 .. 229
      condX(23) // 230 .. 239
      arr[240] != x &&
      arr[241] != x &&
      arr[242] != x &&
      arr[243] != x &&
      arr[244] != x &&
      arr[245] != x &&
      arr[246] != x &&
      arr[247] != x &&
      arr[248] != x &&
      arr[249] != x
      ) {
      clang_analyzer_value(x); // expected-warning {{{ [250, 255] }}}
      if (
      arr[250] != x &&
      arr[251] != x &&
      //skip arr[252]
      arr[253] != x &&
      arr[254] != x &&
      arr[255] != x
      ) {
        clang_analyzer_value(x); // expected-warning {{32s:252}}
        if (arr[252] != x) {
          clang_analyzer_warnIfReached(); // unreachable
        }
      }
    }
  }
}
