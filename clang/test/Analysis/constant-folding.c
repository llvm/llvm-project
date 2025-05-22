// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify -analyzer-config eagerly-assume=false %s

#define UINT_MAX (~0U)
#define INT_MAX (int)(UINT_MAX & (UINT_MAX >> 1))
#define INT_MIN (int)(UINT_MAX & ~(UINT_MAX >> 1))

void clang_analyzer_eval(int);

// There should be no warnings unless otherwise indicated.

void testComparisons (int a) {
  // Sema can already catch the simple comparison a==a,
  // since that's usually a logic error (and not path-dependent).
  int b = a;
  clang_analyzer_eval(b == a); // expected-warning{{TRUE}}
  clang_analyzer_eval(b >= a); // expected-warning{{TRUE}}
  clang_analyzer_eval(b <= a); // expected-warning{{TRUE}}
  clang_analyzer_eval(b != a); // expected-warning{{FALSE}}
  clang_analyzer_eval(b > a); // expected-warning{{FALSE}}
  clang_analyzer_eval(b < a); // expected-warning{{FALSE}}
}

void testSelfOperations (int a) {
  clang_analyzer_eval((a|a) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a&a) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a^a) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval((a-a) == 0); // expected-warning{{TRUE}}
}

void testIdempotent (int a) {
  clang_analyzer_eval((a*1) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a/1) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a+0) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a-0) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a<<0) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a>>0) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a^0) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a&(~0)) == a); // expected-warning{{TRUE}}
  clang_analyzer_eval((a|0) == a); // expected-warning{{TRUE}}
}

void testReductionToConstant (int a) {
  clang_analyzer_eval((a*0) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval((a&0) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval((a|(~0)) == (~0)); // expected-warning{{TRUE}}
}

void testSymmetricIntSymOperations (int a) {
  clang_analyzer_eval((2+a) == (a+2)); // expected-warning{{TRUE}}
  clang_analyzer_eval((2*a) == (a*2)); // expected-warning{{TRUE}}
  clang_analyzer_eval((2&a) == (a&2)); // expected-warning{{TRUE}}
  clang_analyzer_eval((2^a) == (a^2)); // expected-warning{{TRUE}}
  clang_analyzer_eval((2|a) == (a|2)); // expected-warning{{TRUE}}
}

void testAsymmetricIntSymOperations (int a) {
  clang_analyzer_eval(((~0) >> a) == (~0)); // expected-warning{{TRUE}}
  clang_analyzer_eval((0 >> a) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval((0 << a) == 0); // expected-warning{{TRUE}}

  // Unsigned right shift shifts in zeroes.
  clang_analyzer_eval(((~0U) >> a) != (~0U)); // expected-warning{{UNKNOWN}}
}

void testLocations (char *a) {
  char *b = a;
  clang_analyzer_eval(b == a); // expected-warning{{TRUE}}
  clang_analyzer_eval(b >= a); // expected-warning{{TRUE}}
  clang_analyzer_eval(b <= a); // expected-warning{{TRUE}}
  clang_analyzer_eval(b != a); // expected-warning{{FALSE}}
  clang_analyzer_eval(b > a); // expected-warning{{FALSE}}
  clang_analyzer_eval(b < a); // expected-warning{{FALSE}}
}

void testMixedTypeComparisons (char a, unsigned long b) {
  if (a != 0) return;
  if (b != 0x100) return;

  clang_analyzer_eval(a <= b); // expected-warning{{TRUE}}
  clang_analyzer_eval(b >= a); // expected-warning{{TRUE}}
  clang_analyzer_eval(a != b); // expected-warning{{TRUE}}
}

void testBitwiseRules(unsigned int a, int b, int c) {
  clang_analyzer_eval((a | 1) >= 1);   // expected-warning{{TRUE}}
  clang_analyzer_eval((a | -1) >= -1); // expected-warning{{TRUE}}
  clang_analyzer_eval((a | 2) >= 2);   // expected-warning{{TRUE}}
  clang_analyzer_eval((a | 5) >= 5);   // expected-warning{{TRUE}}
  clang_analyzer_eval((a | 10) >= 10); // expected-warning{{TRUE}}

  // Argument order should not influence this
  clang_analyzer_eval((1 | a) >= 1); // expected-warning{{TRUE}}

  clang_analyzer_eval((a & 1) <= 1);    // expected-warning{{TRUE}}
  clang_analyzer_eval((a & 1) >= 0);    // expected-warning{{TRUE}}
  clang_analyzer_eval((a & 2) <= 2);    // expected-warning{{TRUE}}
  clang_analyzer_eval((a & 5) <= 5);    // expected-warning{{TRUE}}
  clang_analyzer_eval((a & 10) <= 10);  // expected-warning{{TRUE}}
  clang_analyzer_eval((a & -10) <= 10); // expected-warning{{UNKNOWN}}

  // Again, check for different argument order.
  clang_analyzer_eval((1 & a) <= 1); // expected-warning{{TRUE}}

  unsigned int d = a;
  d |= 1;
  clang_analyzer_eval((d | 0) == 0); // expected-warning{{FALSE}}

  // Rules don't apply to signed typed, as the values might be negative.
  clang_analyzer_eval((b | 1) > 0); // expected-warning{{UNKNOWN}}

  // Even for signed values, bitwise OR with a non-zero is always non-zero.
  clang_analyzer_eval((b | 1) == 0);  // expected-warning{{FALSE}}
  clang_analyzer_eval((b | -2) == 0); // expected-warning{{FALSE}}
  clang_analyzer_eval((b | 10) == 0); // expected-warning{{FALSE}}
  clang_analyzer_eval((b | 0) == 0);  // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((b | -2) >= 0); // expected-warning{{FALSE}}

  // Check that we can operate with negative ranges
  if (b < 0) {
    clang_analyzer_eval((b | -1) == -1);   // expected-warning{{TRUE}}
    clang_analyzer_eval((b | -10) >= -10); // expected-warning{{TRUE}}
    clang_analyzer_eval((b & 0) == 0);     // expected-warning{{TRUE}}
    clang_analyzer_eval((b & -10) <= -10); // expected-warning{{TRUE}}
    clang_analyzer_eval((b & 5) >= 0);     // expected-warning{{TRUE}}

    int e = (b | -5);
    clang_analyzer_eval(e >= -5 && e <= -1); // expected-warning{{TRUE}}

    if (b < -20) {
      clang_analyzer_eval((b | e) >= -5);    // expected-warning{{TRUE}}
      clang_analyzer_eval((b & -10) < -20);  // expected-warning{{TRUE}}
      clang_analyzer_eval((b & e) < -20);    // expected-warning{{TRUE}}
      clang_analyzer_eval((b & -30) <= -30); // expected-warning{{TRUE}}

      if (c >= -30 && c <= -10) {
        clang_analyzer_eval((b & c) <= -20); // expected-warning{{TRUE}}
      }
    }

    if (a <= 40) {
      int g = (int)a & b;
      clang_analyzer_eval(g <= 40 && g >= 0); // expected-warning{{TRUE}}
    }

    // Check that we can reason about the result even if know nothing
    // about one of the operands.
    clang_analyzer_eval((b | c) != 0); // expected-warning{{TRUE}}
  }

  if (a <= 30 && b >= 10 && c >= 20) {
    // Check that we can reason about non-constant operands.
    clang_analyzer_eval((b | c) >= 20); // expected-warning{{TRUE}}

    // Check that we can reason about the resulting range even if
    // the types are not the same, but we still can convert operand
    // ranges.
    clang_analyzer_eval((a | b) >= 10); // expected-warning{{TRUE}}
    clang_analyzer_eval((a & b) <= 30); // expected-warning{{TRUE}}

    if (b <= 20) {
      clang_analyzer_eval((a & b) <= 20); // expected-warning{{TRUE}}
    }
  }

  // Check that dynamically computed constants also work.
  unsigned int constant = 1 << 3;
  unsigned int f = a | constant;
  clang_analyzer_eval(f >= constant); // expected-warning{{TRUE}}

  // Check that nested expressions also work.
  clang_analyzer_eval(((a | 10) | 5) >= 10); // expected-warning{{TRUE}}

  if (a < 10) {
    clang_analyzer_eval((a | 20) >= 20); // expected-warning{{TRUE}}
  }

  if (a > 10) {
    clang_analyzer_eval((a & 1) <= 1); // expected-warning{{TRUE}}
  }
}

unsigned reset(void);

void testCombinedSources(unsigned a, unsigned b) {
  if (b >= 10 && (a | b) <= 30) {
    // Check that we can merge constraints from (a | b), a, and b.
    // Because of the order of assumptions, we already know that (a | b) is [10, 30].
    clang_analyzer_eval((a | b) >= 10 && (a | b) <= 30); // expected-warning{{TRUE}}
  }

  a = reset();
  b = reset();

  if ((a | b) <= 30 && b >= 10) {
    // Check that we can merge constraints from (a | b), a, and b.
    // At this point, we know that (a | b) is [0, 30], but the knowledge
    // of b >= 10 added later can help us to refine it and change it to [10, 30].
    clang_analyzer_eval(10 <= (a | b) && (a | b) <= 30); // expected-warning{{TRUE}}
  }

  a = reset();
  b = reset();

  unsigned c = (a | b) & (a != b);
  if (c <= 40 && a == b) {
    // Even though we have a directo constraint for c [0, 40],
    // we can get a more precise range by looking at the expression itself.
    clang_analyzer_eval(c == 0); // expected-warning{{TRUE}}
  }
}

void testRemainderRules(unsigned int a, unsigned int b, int c, int d) {
  // Check that we know that remainder of zero divided by any number is still 0.
  clang_analyzer_eval((0 % c) == 0); // expected-warning{{TRUE}}

  clang_analyzer_eval((10 % a) <= 10); // expected-warning{{TRUE}}

  if (a <= 30 && b <= 50) {
    clang_analyzer_eval((40 % a) < 30); // expected-warning{{TRUE}}
    clang_analyzer_eval((a % b) < 50);  // expected-warning{{TRUE}}
    clang_analyzer_eval((b % a) < 30);  // expected-warning{{TRUE}}

    if (a >= 10) {
      // Even though it seems like a valid assumption, it is not.
      // Check that we are not making this mistake.
      clang_analyzer_eval((a % b) >= 10); // expected-warning{{UNKNOWN}}

      // Check that we can we can infer when remainder is equal
      // to the dividend.
      clang_analyzer_eval((4 % a) == 4); // expected-warning{{TRUE}}
      if (b < 7) {
        clang_analyzer_eval((b % a) < 7); // expected-warning{{TRUE}}
      }
    }
  }

  if (c > -10) {
    clang_analyzer_eval((d % c) < INT_MAX);     // expected-warning{{TRUE}}
    clang_analyzer_eval((d % c) > INT_MIN + 1); // expected-warning{{TRUE}}
  }

  // Check that we can reason about signed integers when they are
  // known to be positive.
  if (c >= 10 && c <= 30 && d >= 20 && d <= 50) {
    clang_analyzer_eval((5 % c) == 5);  // expected-warning{{TRUE}}
    clang_analyzer_eval((c % d) <= 30); // expected-warning{{TRUE}}
    clang_analyzer_eval((c % d) >= 0);  // expected-warning{{TRUE}}
    clang_analyzer_eval((d % c) < 30);  // expected-warning{{TRUE}}
    clang_analyzer_eval((d % c) >= 0);  // expected-warning{{TRUE}}
  }

  if (c >= -30 && c <= -10 && d >= -20 && d <= 50) {
    // Test positive LHS with negative RHS.
    clang_analyzer_eval((40 % c) < 30);  // expected-warning{{TRUE}}
    clang_analyzer_eval((40 % c) > -30); // expected-warning{{TRUE}}

    // Test negative LHS with possibly negative RHS.
    clang_analyzer_eval((-10 % d) < 50);  // expected-warning{{TRUE}}
    clang_analyzer_eval((-20 % d) > -50); // expected-warning{{TRUE}}

    // Check that we don't make wrong assumptions
    clang_analyzer_eval((-20 % d) > -20); // expected-warning{{UNKNOWN}}

    // Check that we can reason about negative ranges...
    clang_analyzer_eval((c % d) < 50); // expected-warning{{TRUE}}
    /// ...both ways
    clang_analyzer_eval((d % c) < 30); // expected-warning{{TRUE}}

    if (a <= 10) {
      // Result is unsigned.  This means that 'c' is casted to unsigned.
      // We don't want to reason about ranges changing boundaries with
      // conversions.
      clang_analyzer_eval((a % c) < 30); // expected-warning{{UNKNOWN}}
    }
  }

  // Check that we work correctly when minimal unsigned value from a range is
  // equal to the signed minimum for the same bit width.
  unsigned int x = INT_MIN;
  if (a >= x && a <= x + 10) {
    clang_analyzer_eval((b % a) < x + 10); // expected-warning{{TRUE}}
  }
}

void testDisequalityRules(unsigned int u1, unsigned int u2, unsigned int u3,
                          int s1, int s2, int s3, unsigned char uch,
                          signed char sch, short ssh, unsigned short ush) {

  // Checks for overflowing values
  if (u1 > INT_MAX && u1 <= UINT_MAX / 2 + 4 && u1 != UINT_MAX / 2 + 2 &&
      u1 != UINT_MAX / 2 + 3 && s1 >= INT_MIN + 1 && s1 <= INT_MIN + 2) {
    // u1: [INT_MAX+1, INT_MAX+1]U[INT_MAX+4, INT_MAX+4],
    // s1: [INT_MIN+1, INT_MIN+2]
    clang_analyzer_eval(u1 != s1); // expected-warning{{TRUE}}
  }

  if (u1 >= INT_MIN && u1 <= INT_MIN + 2 &&
      s1 > INT_MIN + 2 && s1 < INT_MIN + 4) {
    // u1: [INT_MAX+1, INT_MAX+1]U[INT_MAX+4, INT_MAX+4],
    // s1: [INT_MIN+3, INT_MIN+3]
    clang_analyzer_eval(u1 != s1); // expected-warning{{TRUE}}
  }

  if (s1 < 0 && s1 > -4 && u1 > UINT_MAX - 4 && u1 < UINT_MAX - 1) {
    // s1: [-3, -1], u1: [UINT_MAX - 3, UINT_MAX - 2]
    clang_analyzer_eval(u1 != s1); // expected-warning{{TRUE}}
    clang_analyzer_eval(s1 != u1); // expected-warning{{TRUE}}
  }

  if (s1 < 1 && s1 > -6 && s1 != -4 && s1 != -3 &&
      u1 > UINT_MAX - 4 && u1 < UINT_MAX - 1) {
    // s1: [-5, -5]U[-2, 0], u1: [UINT_MAX - 3, UINT_MAX - 2]
    clang_analyzer_eval(u1 != s1); // expected-warning{{TRUE}}
  }

  if (s1 < 1 && s1 > -7 && s1 != -4 && s1 != -3 &&
      u1 > UINT_MAX - 4 && u1 < UINT_MAX - 1) {
    // s1: [-6, -5]U[-2, 0], u1: [UINT_MAX - 3, UINT_MAX - 2]
    clang_analyzer_eval(u1 != s1); // expected-warning{{TRUE}}
  }

  if (s1 > 4 && u1 < 4) {
    // s1: [4, INT_MAX], u1: [0, 3]
    clang_analyzer_eval(s1 != u1); // expected-warning{{TRUE}}
  }

  // Check when RHS is in between two Ranges in LHS
  if (((u1 >= 1 && u1 <= 2) || (u1 >= 8 && u1 <= 9)) &&
      u2 >= 5 && u2 <= 6) {
    // u1: [1, 2]U[8, 9], u2: [5, 6]
    clang_analyzer_eval(u1 != u2); // expected-warning{{TRUE}}
  }

  // Checks for concrete value with same type
  if (u1 > 1 && u1 < 3 && u2 > 1 && u2 < 3) {
    // u1: [2, 2], u2: [2, 2]
    clang_analyzer_eval(u1 != u2); // expected-warning{{FALSE}}
  }

  // Check for concrete value with different types
  if (u1 > 4 && u1 < 6 && s1 > 4 && s1 < 6) {
    // u1: [5, 5], s1: [5, 5]
    clang_analyzer_eval(u1 != s1); // expected-warning{{FALSE}}
  }

  // Checks when ranges are not overlapping
  if (u1 <= 10 && u2 >= 20) {
    // u1: [0,10], u2: [20,UINT_MAX]
    clang_analyzer_eval(u1 != u2); // expected-warning{{TRUE}}
  }

  if (s1 <= INT_MIN + 10 && s2 >= INT_MAX - 10) {
    // s1: [INT_MIN,INT_MIN + 10], s2: [INT_MAX - 10,INT_MAX]
    clang_analyzer_eval(s1 != s2); // expected-warning{{TRUE}}
  }

  // Checks when ranges are completely overlapping and have more than one point
  if (u1 >= 20 && u1 <= 50 && u2 >= 20 && u2 <= 50) {
    // u1: [20,50], u2: [20,50]
    clang_analyzer_eval(u1 != u2); // expected-warning{{UNKNOWN}}
  }

  if (s1 >= -20 && s1 <= 20 && s2 >= -20 && s2 <= 20) {
    // s1: [-20,20], s2: [-20,20]
    clang_analyzer_eval(s1 != s2); // expected-warning{{UNKNOWN}}
  }

  // Checks when ranges are partially overlapping
  if (u1 >= 100 && u1 <= 200 && u2 >= 150 && u2 <= 300) {
    // u1: [100,200], u2: [150,300]
    clang_analyzer_eval(u1 != u2); // expected-warning{{UNKNOWN}}
  }

  if (s1 >= -80 && s1 <= -50 && s2 >= -100 && s2 <= -75) {
    // s1: [-80,-50], s2: [-100,-75]
    clang_analyzer_eval(s1 != s2); // expected-warning{{UNKNOWN}}
  }

  // Checks for ranges which are subset of one-another
  if (u1 >= 500 && u1 <= 1000 && u2 >= 750 && u2 <= 1000) {
    // u1: [500,1000], u2: [750,1000]
    clang_analyzer_eval(u1 != u2); // expected-warning{{UNKNOWN}}
  }

  if (s1 >= -1000 && s1 <= -500 && s2 >= -750 && s2 <= -500) {
    // s1: [-1000,-500], s2: [-750, -500]
    clang_analyzer_eval(s1 != s2); // expected-warning{{UNKNOWN}}
  }

  // Checks for comparison between different types
  // Using different variables as previous constraints may interfere in the
  // reasoning.
  if (u3 <= 255 && s3 < 0) {
    // u3: [0, 255], s3: [INT_MIN, -1]
    clang_analyzer_eval(u3 != s3); // expected-warning{{TRUE}}
  }

  // Checks for char-uchar types
  if (uch >= 1 && sch <= 1) {
    // uch: [1, UCHAR_MAX], sch: [SCHAR_MIN, 1]
    clang_analyzer_eval(uch != sch); // expected-warning{{UNKNOWN}}
  }

  if (uch > 1 && sch < 1) {
    // uch: [2, UCHAR_MAX], sch: [SCHAR_MIN, 0]
    clang_analyzer_eval(uch != sch); // expected-warning{{TRUE}}
    clang_analyzer_eval(sch != uch); // expected-warning{{TRUE}}
  }

  if (uch <= 1 && uch >= 1 && sch <= 1 && sch >= 1) {
    // uch: [1, 1], sch: [1, 1]
    clang_analyzer_eval(uch != sch); // expected-warning{{FALSE}}
  }

  // Checks for short-ushort types
  if (ush >= 1 && ssh <= 1) {
    // ush: [1, USHRT_MAX], ssh: [SHRT_MIN, 1]
    clang_analyzer_eval(ush != ssh); // expected-warning{{UNKNOWN}}
  }

  if (ush > 1 && ssh < 1) {
    // ush: [2, USHRT_MAX], ssh: [SHRT_MIN, 0]
    clang_analyzer_eval(ush != ssh); // expected-warning{{TRUE}}
  }

  if (ush <= 1 && ush >= 1 && ssh <= 1 && ssh >= 1) {
    // ush: [1, 1], ssh: [1, 1]
    clang_analyzer_eval(ush != ssh); // expected-warning{{FALSE}}
  }
}
