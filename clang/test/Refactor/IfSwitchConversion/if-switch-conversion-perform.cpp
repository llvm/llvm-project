void foo();

void simpleCompoundBodyIf(int x) {
  foo();

  if (x == 2) { // CHECK1: "switch (" [[@LINE]]:3 -> [[@LINE]]:7
    (void)x;    // CHECK1-NEXT: ") {\ncase " [[@LINE-1]]:8 -> [[@LINE-1]]:12
                // CHECK1-NEXT: ":" [[@LINE-2]]:13 -> [[@LINE-2]]:16
  } else {      // CHECK1-NEXT: "break;\ndefault:" [[@LINE]]:3 -> [[@LINE]]:11
    foo();
  }

  if (((x) == 22 || x == 3) || x == 4) { // CHECK1: "switch (" [[@LINE]]:3 -> [[@LINE]]:9
    // CHECK1-NEXT: ") {\ncase " [[@LINE-1]]:10 -> [[@LINE-1]]:15
    // CHECK1-NEXT: ":\ncase " [[@LINE-2]]:17 -> [[@LINE-2]]:26
    // CHECK1-NEXT: ":\ncase " [[@LINE-3]]:27 -> [[@LINE-3]]:37
    // CHECK1-NEXT: ":" [[@LINE-4]]:38 -> [[@LINE-4]]:41
  } else {  // CHECK1-NEXT: "break;\ndefault:" [[@LINE]]:3 -> [[@LINE]]:11
  }

  foo();
}

// RUN: clang-refactor-test perform -action if-switch-conversion -at=%s:6:3 -at=%s:13:3 %s | FileCheck --check-prefix=CHECK1 %s

void colonInsertionCompoundBody(int x) {
  if (x == 2) // CHECK2: ":" [[@LINE]]:13 -> [[@LINE+2]]:4

  {

  }
  else {
  }
}

// RUN: clang-refactor-test perform -action if-switch-conversion -at=%s:27:3 %s | FileCheck --check-prefix=CHECK2 %s

void colonInsertionNonCompoundBody(int x) {
  if (x == 2) foo(); // CHECK3: ":" [[@LINE]]:13 -> [[@LINE]]:14
  else {
  }

  if (x == 2) // CHECK3: ":" [[@LINE]]:13 -> [[@LINE]]:14
    foo();
  else {
  }

  if ((x == (2)) /*comment*/) // CHECK3: ":" [[@LINE]]:15 -> [[@LINE]]:30
    foo();
  else {
  }

  if (x == 2 // CHECK3: ":" [[@LINE]]:13 -> [[@LINE+1]]:8
      )
    // comment
    foo();
  else {
  }
}

// RUN: clang-refactor-test perform -action if-switch-conversion -at=%s:39:3 -at=%s:43:3 -at=%s:48:3 -at=%s:53:3 %s | FileCheck --check-prefix=CHECK3 %s

void colonInsertionFailure(int x) {
#define EMPTY_MACRO
  if (x == 1 EMPTY_MACRO ) foo();
  else {
  }
}

// RUN: not clang-refactor-test perform -action if-switch-conversion -at=%s:65:3 %s 2>&1 | FileCheck --check-prefix=CHECK-ERR1 %s
// CHECK-ERR1: failed to perform the refactoring operation (couldn't find the location of ')')

void elseNonCompoundBody(int x) {
#ifdef WITH_ELSEIF
  if (x == 1) foo(); else
#endif
  if (x == 2)
    foo();
  else // CHECK4: "break;\ndefault:" [[@LINE]]:3 -> [[@LINE]]:7
    foo();

#ifdef WITH_ELSEIF
  if (x == 1) foo(); else
#endif
  if (x == 2)
    foo();
  else foo(); // CHECK4: "break;\ndefault:" [[@LINE]]:3 -> [[@LINE]]:7

#ifdef WITH_ELSEIF
  if (x == 1) foo(); else
#endif
  if (x == 2) foo(); /*comment*/ else foo(); // CHECK4: "\nbreak;\ndefault:" [[@LINE]]:34 -> [[@LINE]]:38

#ifdef WITH_ELSEIF
  if (x == 1) foo(); else
#endif
  if (x == 2) ; else ; // CHECK4: "\nbreak;\ndefault:" [[@LINE]]:17 -> [[@LINE]]:21
}

void elseCompoundBody(int x) {
#ifdef WITH_ELSEIF
  if (x == 1) foo(); else
#endif
  if (x == 2) { foo(); } else { foo(); } // CHECK4: "\nbreak;\ndefault:" [[@LINE]]:24 -> [[@LINE]]:32

#ifdef WITH_ELSEIF
  if (x == 1) foo(); else
#endif
  if (x == 2) {
    // comment.
  }
  else // CHECK4: "break;\ndefault:" [[@LINE-1]]:3 -> [[@LINE+1]]:4
  {
    foo();
  }
}

// RUN: clang-refactor-test perform -action if-switch-conversion -at=%s:77:3 -at=%s:85:3 -at=%s:92:3 -at=%s:97:3 -at=%s:104:3 -at=%s:109:3 %s | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test perform -action if-switch-conversion -at=%s:77:3 -at=%s:85:3 -at=%s:92:3 -at=%s:97:3 -at=%s:104:3 -at=%s:109:3 %s -D WITH_ELSEIF | FileCheck --check-prefix=CHECK4 %s

void elseIfCompoundBody(int x) {
  if (x == 2) {
    foo();
  } else if (x == 3) { // CHECK5: "break;\ncase " [[@LINE]]:3 -> [[@LINE]]:19
    foo();             // CHECK5-NEXT: ":" [[@LINE-1]]:20 -> [[@LINE-1]]:23
  } else if (x == 4 || x == 55) { // CHECK5-NEXT: "break;\ncase " [[@LINE]]:3 -> [[@LINE]]:19
    foo(); // CHECK5-NEXT: ":\ncase " [[@LINE-1]]:20 -> [[@LINE-1]]:29
           // CHECK5-NEXT: ":" [[@LINE-2]]:31 -> [[@LINE-2]]:34
  }

  if (x == 2) { foo(); } else if (x == 3) { foo(); } // CHECK5: "\nbreak;\ncase " [[@LINE]]:24 -> [[@LINE]]:40
  // CHECK5-NEXT: ":" [[@LINE-1]]:41 -> [[@LINE-1]]:44

  if (x == 2) {
    // comment.
  }
  else if (x == 3) // CHECK5: "break;\ncase " [[@LINE-1]]:3 -> [[@LINE]]:17
  {                // CHECK5-NEXT: ":" [[@LINE-1]]:18 -> [[@LINE]]:4
    foo();
  }
}

void elseIfNonCompoundBody(int x) {
  if (x == 2)
    foo();
  else if (x == 21) // CHECK5: "break;\ncase " [[@LINE]]:3 -> [[@LINE]]:17
    foo();          // CHECK5-NEXT: ":" [[@LINE-1]]:19 -> [[@LINE-1]]:20
  else if (x == 5) ;// CHECK5-NEXT: "break;\ncase " [[@LINE]]:3 -> [[@LINE]]:17
                    // CHECK5-NEXT: ":" [[@LINE-1]]:18 -> [[@LINE-1]]:19

  if (x == 2) foo(); /*comment*/ else if (x == 3) foo(); // CHECK5: "\nbreak;\ncase " [[@LINE]]:34 -> [[@LINE]]:48
  // CHECK5-NEXT: ":" [[@LINE-1]]:49 -> [[@LINE-1]]:50

  if (x == 2) ; else if (x == 3) ; // CHECK5: "\nbreak;\ncase " [[@LINE]]:17 -> [[@LINE]]:31
}

// RUN: clang-refactor-test perform -action if-switch-conversion -at=%s:122:3 -at=%s:131:3 -at=%s:134:3 -at=%s:144:3 -at=%s:151:3 -at=%s:154:3 %s | FileCheck --check-prefix=CHECK5 %s

void closingBraceInsertion(int x) {
  if (x == 2) {
  } else if (x == 3) {
  } else {
    foo();
  } // CHECK6: "break;\n" [[@LINE]]

  if (x == 3) {
  } else if (x == 4) {
  } // CHECK6: "break;\n" [[@LINE]]

  if (x == 2)
    foo();
  else if (x == 3)
    foo();
  else if (x == 4) // CHECK6: "\nbreak;\n}" [[@LINE+1]]:11 -> [[@LINE+1]]:11
    foo();

  if (x == 2)
    foo();
  else // CHECK6: "\nbreak;\n}" [[@LINE+1]]:11 -> [[@LINE+1]]:11
    foo();

  if (x == 2) foo(); // CHECK6: "\nbreak;\n}" [[@LINE+1]]:35 -> [[@LINE+1]]:35
  else foo(); // preserve comments

  if (x == 2) foo();
  else if (x == 3) // CHECK6: "\nbreak;\n}" [[@LINE+1]]:12 -> [[@LINE+1]]:12
    foo() ; foo(); // no preserve

  if (x == 2) foo(); // CHECK6: "\nbreak;\n}" [[@LINE+1]]:11 -> [[@LINE+1]]:11
  else ; ;
}

// RUN: clang-refactor-test perform -action if-switch-conversion -at=%s:160:3 -at=%s:166:3 -at=%s:170:3 -at=%s:177:3 -at=%s:182:3 -at=%s:185:3 -at=%s:189:3 %s | FileCheck --check-prefix=CHECK6 %s

void needBreaks(int x) {
  if (x == 2) {
    return;
    x = 3;
  } else if (x == 3) { // CHECK7: "break;\ncase " [[@LINE]]
    foo();
    return;
    foo();
  } else { // CHECK7: "break;\ndefault:" [[@LINE]]
    if (x == 1) {
      return;
    }
  } // CHECK7: "break;\n" [[@LINE]]:3

  if (x == 2)
    if (x == 3)
      return;
    else ; else // CHECK7: "\nbreak;\ndefault:" [[@LINE]]
    while (x < 2)
      return; // CHECK7: "\nbreak;\n}" [[@LINE]]:52
}

void noNeedForBreaks(int x) {
  if (x == 2) {
    return;
  } else if (x == 3) { // CHECK7: "case " [[@LINE]]
    foo();
    return;
  } else { // CHECK7: "default:" [[@LINE]]
    if (x == 1) {
    }
    {
      return;
    }
  } // CHECK7-NOT: "{{.*}}break{{.*}}" [[@LINE]]

  if (x == 2) return; else return; // CHECK7: "\ndefault:" [[@LINE]]
  // CHECK7: "\n}" [[@LINE-1]]

  // Invalid returns should work as well.
  if (x == 2)
    return 1;
  else        // CHECK7: "default:" [[@LINE]]
    return 2; // CHECK7: "\n}" [[@LINE]]
}

int noNeedForBreaksInvalidRets(int x) {
  if (x == 2)
    return; // This omits the 'break'.
  // But this doesn't (should it?).
  else { // CHECK7: "default:" [[@LINE]]
    return "";
  } // CHECK7: "break;\n" [[@LINE]]
}

// RUN: clang-refactor-test perform -action if-switch-conversion -at=%s:196:3 -at=%s:209:3 -at=%s:218:3 -at=%s:231:3 -at=%s:235:3 -at=%s:242:3 %s | FileCheck --check-prefix=CHECK7 %s

void needBraces(int x) {
  if (x == 2) { // CHECK8: ": {" [[@LINE]]:13 -> [[@LINE]]:16
    int a = x;
  } else if (x == 1) { // CHECK8-NEXT: "break;\n}\ncase " [[@LINE]]:3
    int a = 0, y = 1;  // CHECK8-NEXT: ": {" [[@LINE-1]]:20 -> [[@LINE-1]]:23
    return;
  } else if (x == 3 || x == 4) { // CHECK8-NEXT: "}\ncase " [[@LINE]]:3
    int m = 2;                   // CHECK8-NEXT: ":\ncase " [[@LINE-1]]
                                 // CHECK8-NEXT: ": {" [[@LINE-2]]:30 -> [[@LINE-2]]:33
  } else if (x == 5) { // CHECK8-NEXT: "break;\n}\ncase " [[@LINE]]:3
    return; // CHECK8-NEXT: ":" [[@LINE-1]]:20 -> [[@LINE-1]]:23
  } else {  // CHECK8-NEXT: "default: {" [[@LINE]]:3 -> [[@LINE]]:11
    foo();
    int k = x;
    foo();
  } // CHECK8-NEXT: "break;\n}\n" [[@LINE]]:3 -> [[@LINE]]:3

  if (x == 2) { // CHECK8: ": {" [[@LINE]]
    int a = 2;
  } else if (x == 3) { // CHECK8: "break;\n}\ncase " [[@LINE]]
    int b = x;         // CHECK8-NEXT: ": {" [[@LINE-1]]
  } // CHECK8-NEXT: "break;\n}\n" [[@LINE]]

  if (x == 2) // CHECK8: ": {" [[@LINE]]:13 -> [[@LINE]]:14
    int a = x;
  else if (x == 1 || x == 3) // CHECK8-NEXT: "break;\n}\ncase " [[@LINE]]:3 -> [[@LINE]]:17
    int b = 2; // CHECK8-NEXT: ":\ncase " [[@LINE-1]]
               // CHECK8-NEXT: ": {" [[@LINE-2]]:28 -> [[@LINE-2]]:29
  else if (x == 4) // CHECK8-NEXT: "break;\n}\ncase " [[@LINE]]:3 -> [[@LINE]]:17
    foo();         // CHECK8-NEXT: ":" [[@LINE-1]]
  else if (x == 5) // CHECK8-NEXT: "break;\ncase " [[@LINE]]
    return;        // CHECK8-NEXT: ":" [[@LINE-1]]
  else             // CHECK8-NEXT: "default: {" [[@LINE]]:3 -> [[@LINE]]:7
    int c = x;
  // CHECK8-NEXT: "\nbreak;\n}\n}" [[@LINE-1]]:15 -> [[@LINE-1]]:15

  if (x == 2) int a = 1; else int k = x;
  // CHECK8: ": {" [[@LINE-1]]:13 -> [[@LINE-1]]:14
  // CHECK8-NEXT: "\nbreak;\n}\ndefault: {" [[@LINE-2]]:26 -> [[@LINE-2]]:30
  // CHECK8-NEXT: "\nbreak;\n}\n}" [[@LINE-3]]:41 -> [[@LINE-3]]:41
}

void noBracesNeeded(int x) {
  if (x == 2) { // CHECK8: ":" [[@LINE]]
    if (int *z = p) {
    }
  } else if (x == 3) {  // CHECK8: "break;\ncase " [[@LINE]]
    for (int z = 0; z < x ; ++z) ; // CHECK8: ":" [[@LINE-1]]
  } else if (x == 4) { // CHECK8: "break;\ncase " [[@LINE]]
    { // CHECK8: ":" [[@LINE-1]]
      int a = 1;
    }
  }
}

// RUN: clang-refactor-test perform -action if-switch-conversion -at=%s:253:3 -at=%s:269:3 -at=%s:275:3 -at=%s:288:3 -at=%s:295:3 %s | FileCheck --check-prefix=CHECK8 %s

#define MACRO(X) X

void macroArg(int x) {
  // macro-arg: +1:9
  MACRO(if (x == 2) { // MACRO-ARG: "switch (" [[@LINE]]:9 -> [[@LINE]]:13
    ;                 // MACRO-ARG: ") {\ncase " [[@LINE-1]]:14 -> [[@LINE-1]]:18
                      // MACRO-ARG: ":" [[@LINE-2]]:19 -> [[@LINE-2]]:22
  } else if (x == 3) { // MACRO-ARG: "break;\ncase " [[@LINE]]:3 -> [[@LINE]]:19
    ;                 // MACRO-ARG: ":" [[@LINE-1]]:20 -> [[@LINE-1]]:23
  }); // MACRO-ARG: "break;\n" [[@LINE]]:3 -> [[@LINE]]:3
}

// RUN: clang-refactor-test perform -action if-switch-conversion -at=macro-arg %s | FileCheck --check-prefix=MACRO-ARG %s
