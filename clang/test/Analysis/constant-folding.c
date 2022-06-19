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
