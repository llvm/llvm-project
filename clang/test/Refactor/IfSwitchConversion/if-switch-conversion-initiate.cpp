void foo(); void foobar();

void simpleCompoundBodyIf(int x) {
  foo();

  if (x == 2) {
    int y = x;
  } else if (x == 3) {
    foo();
  } else {
    foobar();
  }

  foobar();
}

// RUN: clang-refactor-test list-actions -at=%s:6:3 %s | FileCheck --check-prefix=CHECK-ACTION %s
// CHECK-ACTION: Convert to Switch

// Ensure the the action can be initiated around the ifs:

// RUN: clang-refactor-test initiate -action if-switch-conversion -in=%s:6:3-15 -in=%s:8:3-22 -in=%s:10:3-10 %s | FileCheck --check-prefix=CHECK1 %s
// CHECK1: Initiated the 'if-switch-conversion' action at 6:3

// Ensure that the action can't be initiated when not around the ifs:

// RUN: not clang-refactor-test initiate -action if-switch-conversion -in=%s:1:1-10 -in=%s:3:1-18 -in=%s:4:1-9 -in=%s:6:1-2 -in=%s:14:1-12 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s
// CHECK-NO: Failed to initiate the refactoring action
// CHECK-NO-NOT: Initiated the 'if-switch-conversion' action

// Ensure that the action can't be initiated inside the ifs:
// RUN: not clang-refactor-test initiate -action if-switch-conversion -in=%s:7:1-15 -in=%s:9:1-11 -in=%s:11:1-14 -in=%s:12:1-4 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

void nestedIf(int x) {
  if (x == 2) {
    if (x == 3) {
      foo();
    } else {
      foo();
    }
  } else {
    foobar();
  }
}

// RUN: clang-refactor-test initiate -action if-switch-conversion -in=%s:35:3-15 -in=%s:35:3-10 %s | FileCheck --check-prefix=CHECK2 %s
// CHECK2: Initiated the 'if-switch-conversion' action at 35:3

// RUN: clang-refactor-test initiate -action if-switch-conversion -in=%s:36:5-17 -in=%s:38:5-12 %s | FileCheck --check-prefix=CHECK3 %s
// CHECK3: Initiated the 'if-switch-conversion' action at 36:5





void simpleFlatBodyIfs(int x) {
  if (x == 2)
    foo();
  else if (x == 3)
    foo();

  else if (x == 4) foobar();

  else foo();

  if (x == 2)  foobar();
  else
    //comment
    foo();
}

// RUN: clang-refactor-test initiate -action if-switch-conversion -in=%s:57:3-14 -in=%s:59:3-19 -in=%s:62:3-19 -in=%s:64:3-7 %s | FileCheck --check-prefix=CHECK4 %s
// CHECK4: Initiated the 'if-switch-conversion' action at 57:3

// RUN: not clang-refactor-test initiate -action if-switch-conversion -in=%s:57:1-2 -in=%s:58:1-11 -in=%s:59:1-2 -in=%s:60:1-11 -in=%s:61:1-1 -in=%s:62:1-2 -in=%s:62:20-29 -in=%s:63:1-1 -in=%s:64:1-2 -in=%s:64:8-14 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// RUN: clang-refactor-test initiate -action if-switch-conversion -in=%s:66:3-15 -in=%s:67:3-7 -in=%s:68:1-14 %s | FileCheck --check-prefix=CHECK5 %s
// CHECK5: Initiated the 'if-switch-conversion' action at 66:3

void differentLineCompoundIf(int x) {
  if (x == 2)
  {
    foo();
  }

  else if (x == 3)

  {
    foo();
  }
}

// RUN: clang-refactor-test initiate -action if-switch-conversion -in=%s:81:3-14 -in=%s:86:3-19 -in=%s:87:1-1 %s | FileCheck --check-prefix=CHECK6 %s
// CHECK6: Initiated the 'if-switch-conversion' action at 81:3

// RUN: not clang-refactor-test initiate -action if-switch-conversion -in=%s:81:1-2 -in=%s:82:1-4 -in=%s:84:1-4 -in=%s:85:1-1 -in=%s:86:1-2 -in=%s:88:1-4 -in=%s:90:1-4 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

void simpleEmptyIf(int x) {
  if (x == 1) ;
  else if (x == 2) ;
}

// RUN: clang-refactor-test initiate -action if-switch-conversion -in=%s:99:3-14 -in=%s:100:3-19 %s | FileCheck --check-prefix=CHECK7 %s
// CHECK7: Initiated the 'if-switch-conversion' action at 99:3

// RUN: not clang-refactor-test initiate -action if-switch-conversion -in=%s:99:15-16 -in=%s:100:20-21 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

void sameLineIfElse(int x) {
  if (x == 1) foo(); else foo();
  if (x == 2) { foo(); } else if (x == 3) { foo(); }
}

// RUN: clang-refactor-test initiate -action if-switch-conversion -in=%s:109:3-14 -in=%s:109:22-26 %s | FileCheck --check-prefix=CHECK8 %s
// CHECK8: Initiated the 'if-switch-conversion' action at 109:3

// RUN: not clang-refactor-test initiate -action if-switch-conversion -in=%s:109:15-21 -in=%s:109:27-33 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

// RUN: clang-refactor-test initiate -action if-switch-conversion -in=%s:110:3-15 -in=%s:110:24-43 %s | FileCheck --check-prefix=CHECK9 %s
// CHECK9: Initiated the 'if-switch-conversion' action at 110:3

// RUN: not clang-refactor-test initiate -action if-switch-conversion -in=%s:110:16-23 -in=%s:110:44-53 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

void noIfsWithoutElses(int x) {
  if (x == 1) {
    foo();
  }
  if (x == 2) ;
}

// Ifs without any elses shouldn't be allowed:
// RUN: not clang-refactor-test initiate -action if-switch-conversion -in=%s:124:1-16 -in=%s:127:1-16 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s

void noFancyIfs(const int *p) {
  if (const int *x = p) {
  }
  else if (const int *y = p) {
  }

  if (const int *x = p; *x == 2) {
  } else if (const int *y = p; *y == 3) {
  }
}

// RUN: not clang-refactor-test initiate -action if-switch-conversion -in=%s:134:1-26 -in=%s:136:1-31 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s
// RUN: not clang-refactor-test initiate -action if-switch-conversion -in=%s:139:1-35 -in=%s:140:1-42 %s -std=c++1z 2>&1 | FileCheck --check-prefix=CHECK-NO %s

void prohibitBreaksCasesDefaults(int x) {
  while (x != 0) {
    break;
    // Allowed:
    if (x == 1) foo();
    else foo();
    // Not allowed:
    if (x == 2) break;
    else foo();
    if (x == 2) { foo(); }
    else { break; }
    if (x == 2) { foo(); }
    else if (x == 1) { if (x == 2) { break; } }
  }
  switch (x) {
  case 1:
    // Allowed:
    if (x == 1) foo();
    else foo();
    // Not allowed:
    if (x == 2) foo();
    else if (x == 3) {
  case 2:
      foo();
    }
    if (x == 3) foo();
    else {
  default:
      foo();
    }
  }
}

// RUN: clang-refactor-test initiate -action if-switch-conversion -at=%s:151:5 %s | FileCheck --check-prefix=CHECK10 %s
// CHECK10: Initiated the 'if-switch-conversion' action at 151:5

// RUN: not clang-refactor-test initiate -action if-switch-conversion -at=%s:154:5 -at=%s:156:5 -at=%s:158:5 %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID-STATEMENTS %s
// CHECK-INVALID-STATEMENTS: Failed to initiate the refactoring action (if's body contains a 'break'/'default'/'case' statement)

// RUN: clang-refactor-test initiate -action if-switch-conversion -at=%s:164:5 %s | FileCheck --check-prefix=CHECK11 %s
// CHECK11: Initiated the 'if-switch-conversion' action at 164:5

// RUN: not clang-refactor-test initiate -action if-switch-conversion -at=%s:167:5 -at=%s:172:5 %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID-STATEMENTS %s

#ifdef DISALLOWED
  #define DISALLOW(x) x
#else
  #define DISALLOW(x)
#endif

void allowBreaksInNestedLoops(int x) {
  DISALLOW(while (true)) {
  // Allowed:
  if (x == 1) {
    foo();
  } else if (x == 2) {
    while (x != 0) {
      break;
    }
    DISALLOW(break;)
  }

  if (x == 1) {
    foo();
  } else {
    for (int y = 0; y < x; ++y) {
      break;
    }
    DISALLOW(break;)
  }

  if (x == 1) {
    do {
      break;
    } while (x < 10);
    DISALLOW(break;)
  } else {
    foo();
  }

  if (x == 1) {
    do {
      // nested loop.
      while (true) {
      }
      break;
    } while (x < 10);
    DISALLOW(break;)
  } else {
    foo();
  }

  }

  // Still care about cases and defaults in loops:
  switch (x) {
  case 0:
    if (x == 1) {
      while (true) {
  case 1:
      }
    } else {
      foo();
    }
    break;
  }

  switch (x) {
  case 0:
    if (x == 1) {
      while (true) {
  default:
      }
    } else {
      foo();
    }
    break;
  }
}

// RUN: clang-refactor-test initiate -action if-switch-conversion -at=%s:200:3 %s | FileCheck --check-prefix=CHECK-YES %s
// CHECK-YES: Initiated the 'if-switch-conversion' action

// RUN: clang-refactor-test initiate -action if-switch-conversion -at=%s:209:3 %s | FileCheck --check-prefix=CHECK-YES %s
// RUN: clang-refactor-test initiate -action if-switch-conversion -at=%s:218:3 %s | FileCheck --check-prefix=CHECK-YES %s
// RUN: clang-refactor-test initiate -action if-switch-conversion -at=%s:227:3 %s | FileCheck --check-prefix=CHECK-YES %s

// RUN: not clang-refactor-test initiate -action if-switch-conversion -at=%s:200:3 -at=%s:209:3 -at=%s:218:3 -at=%s:227:3 -at=%s:244:5 -at=%s:256:5 %s 2>&1 -D DISALLOWED | FileCheck --check-prefix=CHECK-INVALID-STATEMENTS %s

void allowBreakDefaultCaseInNestedSwitches(int x) {
  DISALLOW(switch (x)) {
  // Allowed:
  if (x == 1) {
    foo();
  } else if (x == 2) {
    switch (x) {
    case 0:
      foo();
    }
    DISALLOW(case 0: ;)
  }

  if (x == 1) {
    foo();
  } else {
    switch (x) {
    default:
      foo();
    }
    DISALLOW(default: ;)
  }

  if (x == 1) {
    switch (x) {
      break;
    }
    DISALLOW(break;)
  } else {
    foo();
  }

  }
}

// RUN: clang-refactor-test initiate -action if-switch-conversion -at=%s:279:3 %s | FileCheck --check-prefix=CHECK-YES %s
// RUN: clang-refactor-test initiate -action if-switch-conversion -at=%s:289:3 %s | FileCheck --check-prefix=CHECK-YES %s
// RUN: clang-refactor-test initiate -action if-switch-conversion -at=%s:299:3 %s | FileCheck --check-prefix=CHECK-YES %s

// RUN: not clang-refactor-test initiate -action if-switch-conversion -at=%s:279:3 -at=%s:289:3 -at=%s:299:3  %s 2>&1 -D DISALLOWED | FileCheck --check-prefix=CHECK-INVALID-STATEMENTS %s
















bool isTrue();

void allowOnlyEqualsOp(int x) {
  if (x != 1) {
  } else {
  }

  if (x == 1) {
  } else if (x > 2) {
  }

  if (x == 3) {
  } else if (x) {
  }

  if (isTrue()) {
  } else {
  }
}

// RUN: not clang-refactor-test initiate -action if-switch-conversion -at=%s:335:3 -at=%s:339:3 -at=%s:343:3 -at=%s:347:3  %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID-COND %s
// CHECK-INVALID-COND: Failed to initiate the refactoring action (unsupported conditional expression)!

void allowEqualsOpInParens(int x) {
  if ((x == 1)) {
  } else {
  }

  if (x == 1) {
  } else if (((x == 2))) {
  }
}

// RUN: clang-refactor-test initiate -action if-switch-conversion -at=%s:356:3 %s | FileCheck --check-prefix=CHECK-YES %s
// RUN: clang-refactor-test initiate -action if-switch-conversion -at=%s:360:3 %s | FileCheck --check-prefix=CHECK-YES %s

enum Switchable {
  A, B
};

struct Struct {
};

bool operator == (const Struct &, int);

void allowSwitchableTypes(int x, bool b, long l, char c, Switchable e,
                          float f, double d, Struct s, int *ip) {
  // Allowed:
  if (b == true) {
  } else {
  }

  if (1 == 1) {
  } else {
  }

  if (l == 4) {
  } else {
  }

  if (e == A) {
  } else {
  }

  if (x == A) {
  } else {
  }

  if (c == 'x') {
  } else {
  }

  // Disallowed:
  if (f == 0) {
  } else {
  }

  if (d == 0) {
  } else {
  }

  if (x == 0) {
  } else if (x == 0.0) {
  }

  if (s == 0) {
  } else {
  }

  if (ip == 0) {
  } else {
  }
}

// RUN: clang-refactor-test initiate -action if-switch-conversion -location-agnostic -at=%s:380:3 -at=%s:384:3 -at=%s:388:3 -at=%s:392:3 -at=%s:396:3 -at=%s:400:3 %s | FileCheck --check-prefix=CHECK-YES %s
// RUN: not clang-refactor-test initiate -action if-switch-conversion -at=%s:405:3 -at=%s:409:3 -at=%s:413:3 -at=%s:417:3 -at=%s:421:3  %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID-COND %s

template<typename T>
void prohibitDependentOperators(T x) {
  if (x == 0) {
  } else {
  }
}

// RUN: not clang-refactor-test initiate -action if-switch-conversion -at=%s:431:3 %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID-COND %s

int integerFunction();

void checkLHSSame(int x, int y) {
  // Allowed:
  if (integerFunction() == 1) {
  } else if (integerFunction() == 2) {
  }

  // Disallowed:
  if (x == 1) {
  } else if (y == 2) {
  }

  if (x == 1) {
  } else if (2 == 2) {
  }

  if (integerFunction() == 1) {
  } else if (x == 2) {
  }
}

// RUN: clang-refactor-test initiate -action if-switch-conversion -location-agnostic -at=%s:442:3 %s | FileCheck --check-prefix=CHECK-YES %s
// RUN: not clang-refactor-test initiate -action if-switch-conversion -at=%s:447:3 -at=%s:451:3 -at=%s:455:3 %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID-COND %s

void checkRHSConstant(int x, int y, Switchable e) {
  // Allowed:
  if (x == (int)A) {
  } else {
  }

  if (e == (Switchable)1) {
  } else {
  }

  // Disallowed:
  if (x == y) {
  } else {
  }

  if (x == 1) {
  } else if (x == integerFunction()) {
  }
}

// RUN: clang-refactor-test initiate -action if-switch-conversion -location-agnostic -at=%s:465:3 -at=%s:469:3 %s | FileCheck --check-prefix=CHECK-YES %s
// RUN: not clang-refactor-test initiate -action if-switch-conversion -at=%s:474:3 -at=%s:478:3 %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID-COND %s

void checkRHSUnique(int x, int y, Switchable e) {
  // Disallowed:
  if (x == 0) {
  } else if (x == 0) {
  }

  if (e == A) {
  } else if (e == (Switchable)0) {
  }
}

// RUN: not clang-refactor-test initiate -action if-switch-conversion -at=%s:474:3 -at=%s:478:3 %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID-COND %s

void allowLHSParens(int x) {
  if ((x) == 0) {
  } else {
  }
}

void allowRHSParens(int x) {
  if (x == (0)) {
  } else {
  }
}

// RUN: clang-refactor-test initiate -action if-switch-conversion -location-agnostic -at=%s:500:3 -at=%s:506:3 %s | FileCheck --check-prefix=CHECK-YES %s

void allowLogicalOr(int x, int y) {
  // Allowed:
  if (x == 0 || x == 1) {
  } else {
  }

  if (x == 0) {
  } else if (x == 1 || x == 2) {
  }

  if (x == (0) || (x == 1)) {
  } else {
  }

  if (x == 0) {
  } else if ((x == 1 || x == 2)) {
  }

  // Disallowed:
  if (x == 0 && x == 1) {
  } else {
  }

  if (x == 0 | x == 1) {
  } else {
  }

  if (x == 0 || isTrue()) {
  } else if (y == 2) {
  }

  if (x == 0 || x == 1) {
  } else if (y == 2) {
  }

  if (x == 0) {
  } else if (x == 1 || x == integerFunction()) {
  }

  if (x == 1) {
  } else if (x == 2 || x == 1) {
  }
}

// RUN: clang-refactor-test initiate -action if-switch-conversion -location-agnostic -at=%s:515:3 -at=%s:519:3 -at=%s:523:3 -at=%s:527:3 %s | FileCheck --check-prefix=CHECK-YES %s
// RUN: not clang-refactor-test initiate -action if-switch-conversion -at=%s:532:3 -at=%s:536:3 -at=%s:540:3 -at=%s:544:3 -at=%s:548:3 -at=%s:552:3 %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID-COND %s

void parenImpCastsLHSEquivalence(int x) {
  if ((x) == 1) {
  } else if (x == 2) {
  }
}

// RUN: clang-refactor-test initiate -action if-switch-conversion -location-agnostic -at=%s:561:3 %s | FileCheck --check-prefix=CHECK-YES %s

// UNSUPPORTED: system-windows
