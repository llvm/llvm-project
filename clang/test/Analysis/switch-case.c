// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);
void clang_analyzer_warnIfReached();

#define INT_MIN 0x80000000
#define INT_MAX 0x7fffffff

// PR16833: Analyzer consumes memory until killed by kernel OOM killer
// while analyzing large case ranges.
void PR16833(unsigned op) {
  switch (op) {
  case 0x02 << 26 ... 0x03 << 26: // Analyzer should not hang here.
    return;
  }
}

void testAdjustment(int t) {
  switch (t + 1) {
  case 2:
    clang_analyzer_eval(t == 1); // expected-warning{{TRUE}}
    break;
  case 3 ... 10:
    clang_analyzer_eval(t > 1);        // expected-warning{{TRUE}}
    clang_analyzer_eval(t + 2 <= 11);  // expected-warning{{TRUE}}
    clang_analyzer_eval(t > 2);        // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(t + 1 == 3);   // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(t + 1 == 10);  // expected-warning{{UNKNOWN}}
    break;
  default:
    clang_analyzer_warnIfReached();    // expected-warning{{REACHABLE}}
  }
}

void testUnknownVal(int value, int mask) {
  // Once ConstraintManager will process '&' and this test will require some changes.
  switch (value & mask) {
    case 1:
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
      break;
    case 3 ... 10:
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
      break;
    default:
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
}

void testSwitchCond(int arg) {
  if (arg > 10) {
    switch (arg) {
    case INT_MIN ... 10:
      clang_analyzer_warnIfReached(); // no-warning
      break;
    case 11 ... 20:
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
      break;
    default:
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
    }

    switch (arg) {
    case INT_MIN ... 9:
      clang_analyzer_warnIfReached();  // no-warning
      break;
    case 10 ... 20:
      clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
      clang_analyzer_eval(arg > 10);   // expected-warning{{TRUE}}
      break;
    default:
      clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
    }
  } // arg > 10
}

void testDefaultUnreachable(int arg) {
  if (arg > 10) {
    switch (arg) {
    case INT_MIN ... 9:
      clang_analyzer_warnIfReached();   // no-warning
      break;
    case 10 ... INT_MAX:
      clang_analyzer_warnIfReached();   // expected-warning{{REACHABLE}}
      clang_analyzer_eval(arg > 10);    // expected-warning{{TRUE}}
      break;
    default:
      clang_analyzer_warnIfReached();   // no-warning
    }
  }
}

void testBranchReachability(int arg) {
  if (arg > 10 && arg < 20) {
    switch (arg) {
    case INT_MIN ... 4:
      clang_analyzer_warnIfReached(); // no-warning
      break;
    case 5 ... 9:
      clang_analyzer_warnIfReached(); // no-warning
      break;
    case 10 ... 15:
      clang_analyzer_warnIfReached();              // expected-warning{{REACHABLE}}
      clang_analyzer_eval(arg > 10 && arg <= 15);  // expected-warning{{TRUE}}
      break;
    default:
      clang_analyzer_warnIfReached(); // no-warning
      break;
    case 17 ... 25:
      clang_analyzer_warnIfReached();              // expected-warning{{REACHABLE}}
      clang_analyzer_eval(arg >= 17 && arg < 20);  // expected-warning{{TRUE}}
      break;
    case 26 ... INT_MAX:
      clang_analyzer_warnIfReached();   // no-warning
      break;
    case 16:
      clang_analyzer_warnIfReached();   // expected-warning{{REACHABLE}}
      clang_analyzer_eval(arg == 16);   // expected-warning{{TRUE}}
      break;
    }
  }
}

void testDefaultBranchRange(int arg) {
  switch (arg) {
  case INT_MIN ... 9:
    clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
    break;
  case 20 ... INT_MAX:
    clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
    clang_analyzer_eval(arg >= 20);  // expected-warning{{TRUE}}
    break;
  default:
    clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
    clang_analyzer_eval(arg == 16);  // expected-warning{{FALSE}}
    clang_analyzer_eval(arg > 9);    // expected-warning{{TRUE}}
    clang_analyzer_eval(arg <= 20);  // expected-warning{{TRUE}}

  case 16:
    clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
  }
}

void testAllUnreachableButDefault(int arg) {
  if (arg < 0) {
    switch (arg) {
    case 0 ... 9:
      clang_analyzer_warnIfReached(); // no-warning
      break;
    case 20 ... INT_MAX:
      clang_analyzer_warnIfReached(); // no-warning
      break;
    default:
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
      break;
    case 16:
      clang_analyzer_warnIfReached(); // no-warning
    }
    clang_analyzer_warnIfReached();   // expected-warning{{REACHABLE}}
  }
}

void testAllUnreachable(int arg) {
  if (arg < 0) {
    switch (arg) {
    case 0 ... 9:
      clang_analyzer_warnIfReached(); // no-warning
      break;
    case 20 ... INT_MAX:
      clang_analyzer_warnIfReached(); // no-warning
      break;
    case 16:
      clang_analyzer_warnIfReached(); // no-warning
    }
    clang_analyzer_warnIfReached();   // expected-warning{{REACHABLE}}
  }
}

void testDifferentTypes(int arg) {
  switch (arg) {
  case -1U ... 400000000LL:
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
      break;
    default:
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
      break;
  }
}

void testDifferentTypes2(unsigned long arg) {
  switch (arg) {
  case 1UL ... 400000000UL:
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
      break;
    default:
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
      break;
  }
}

void testDifferentTypes3(int arg) {
  switch (arg) {
  case 1UL ... 400000000UL:
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
      break;
    default:
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
      break;
  }
}

void testConstant() {
  switch (3) {
  case 1 ... 5:
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
    break;
  default:
    clang_analyzer_warnIfReached(); // no-warning
    break;
  }
}
