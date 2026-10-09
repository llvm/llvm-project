// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection,alpha.core.DanglingPtrDeref\
// RUN:  -analyzer-config cfg-lifetime=true -analyzer-output=text -verify %s

void clang_analyzer_warnIfReached();

// Enabling CFG lifetime-end elements should not cause regression in the
// path notes. The loop exit path note must point to the next executed
// statmeent and not back to the loop.
void testPathNotesWithLoopScopeEnd() {
  int *p = nullptr;
  for (int i = 0; i < 3; ++i) {
  // expected-note@-1 3 {{Loop condition is true.  Entering loop body}}
  // expected-note@-2   {{The value 2 is assigned to 'i'}}
  // expected-note@-3   {{Loop condition is false. Execution continues on line 17}}
    p = &i; // expected-note {{Value assigned to 'p'}}
  } // expected-note {{'i' is destroyed here}}
  *p = 4;
  // expected-warning@-1 {{Use of 'i' after its lifetime ended}}
  // expected-note@-2    {{Use of 'i' after its lifetime ended}}
}

void testPathNotesWithBlockScopeEnd() {
  int *p = nullptr;
  {
    int n = 0;
    while (n < 3) {
    // expected-note@-1 3 {{Loop condition is true.  Entering loop body}}
    // expected-note@-2   {{Loop condition is false. Execution continues on line 33}}
      p = &n; // expected-note {{Value assigned to 'p'}}
      ++n;    // expected-note {{The value 2 is assigned to 'n'}}
    }
  } // expected-note {{'n' is destroyed here}}
  *p = 4;
  // expected-warning@-1 {{Use of 'n' after its lifetime ended}}
  // expected-note@-2    {{Use of 'n' after its lifetime ended}}
}

void testPathNotesWithWarnIfReached() {
  {
    int i = 0;
    while (i < 3) {
    // expected-note@-1 3 {{Loop condition is true.  Entering loop body}}
    // expected-note@-2   {{Loop condition is false. Execution continues on line 47}} 
      ++i;
    }
  }
  clang_analyzer_warnIfReached();
  // expected-warning@-1 {{REACHABLE}}
  // expected-note@-2    {{REACHABLE}}
}
