// RUN: %clang_analyze_cc1 -w -analyzer-checker=core,debug.ExprInspection\
// RUN:                    -analyzer-config eagerly-assume=false -verify %s

void clang_analyzer_eval(int);

void testAnd(int i, int *p) {
  int *nullP = 0;
  int *knownP = &i;
  clang_analyzer_eval((knownP && knownP) == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval((knownP && nullP) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval((knownP && p) == 1); // expected-warning{{UNKNOWN}}
}

void testOr(int i, int *p) {
  int *nullP = 0;
  int *knownP = &i;
  clang_analyzer_eval((nullP || knownP) == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval((nullP || nullP) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval((nullP || p) == 1); // expected-warning{{UNKNOWN}}
}


// PR13461
int testTypeIsInt(int i, void *p) {
  if (i | (p && p))
    return 1;
  return 0;
}

// These crashed the analyzer at some point.
int between(char *x) {
  extern char start[];
  extern char end[];
  return x >= start && x < end;
}

int undef(void) {}
void useUndef(void) { 0 || undef(); }

void testPointer(void) { (void) (1 && testPointer && 0); }

char *global_ap, *global_bp, *global_cp;
void ambiguous_backtrack_1() {
  for (;;) {
    (global_bp - global_ap ? global_cp[global_bp - global_ap] : 0) || 1;
    global_bp++;
  }
}

int global_a, global_b;
void ambiguous_backtrack_2(int x) {
  global_a = x >= 2 ? 1 : x;
  global_b == x && 9 || 2;
}
