// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core.BoolAssignment,alpha.security.taint -analyzer-store=region -verify -std=c99 -Dbool=_Bool %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core.BoolAssignment,alpha.security.taint -analyzer-store=region -verify -x c++ %s

// Test C++'s bool and C's _Bool.
// FIXME: We stopped warning on these when SValBuilder got smarter about
// casts to bool. Arguably, however, these conversions are okay; the result
// is always 'true' or 'false'.

void test_stdbool_initialization(int y) {
  bool constant = 2; // no-warning
  if (y < 0) {
    bool x = y; // no-warning
    return;
  }
  if (y > 1) {
    bool x = y; // no-warning
    return;
  }
  bool x = y; // no-warning
}

void test_stdbool_assignment(int y) {
  bool x = 0; // no-warning
  if (y < 0) {
    x = y; // no-warning
    return;
  }
  if (y > 1) {
    x = y; // no-warning
    return;
  }
  x = y; // no-warning
}

// Test Objective-C's BOOL

typedef signed char BOOL;

void test_BOOL_initialization(int y) {
  BOOL constant = 2; // expected-warning {{Assignment of a non-Boolean value}}
  if (y < 0) {
    BOOL x = y; // expected-warning {{Assignment of a non-Boolean value}}
    return;
  }
  if (y > 200 && y < 250) {
#ifdef ANALYZER_CM_Z3
    BOOL x = y; // expected-warning {{Assignment of a non-Boolean value}}
#else
    BOOL x = y; // no-warning
#endif
    return;
  }
  if (y >= 127 && y < 150) {
    BOOL x = y; // expected-warning{{Assignment of a non-Boolean value}}
    return;
  }
  if (y > 1) {
    BOOL x = y; // expected-warning {{Assignment of a non-Boolean value}}
    return;
  }
  BOOL x = y; // no-warning
}

void test_BOOL_assignment(int y) {
  BOOL x = 0; // no-warning
  if (y < 0) {
    x = y; // expected-warning {{Assignment of a non-Boolean value}}
    return;
  }
  if (y > 1) {
    x = y; // expected-warning {{Assignment of a non-Boolean value}}
    return;
  }
  x = y; // no-warning
}


// Test MacTypes.h's Boolean

typedef unsigned char Boolean;

void test_Boolean_initialization(int y) {
  Boolean constant = 2; // expected-warning {{Assignment of a non-Boolean value}}
  if (y < 0) {
    Boolean x = y; // expected-warning {{Assignment of a non-Boolean value}}
    return;
  }
  if (y > 1) {
    Boolean x = y; // expected-warning {{Assignment of a non-Boolean value}}
    return;
  }
  Boolean x = y; // no-warning
}

void test_Boolean_assignment(int y) {
  Boolean x = 0; // no-warning
  if (y < 0) {
    x = y; // expected-warning {{Assignment of a non-Boolean value}}
    return;
  }
  if (y > 1) {
    x = y; // expected-warning {{Assignment of a non-Boolean value}}
    return;
  }
  x = y; // no-warning
}

int scanf(const char *format, ...);
void test_tainted_Boolean() {
  int n;
  scanf("%d", &n);
  Boolean copy = n; // expected-warning {{Might assign a tainted non-Boolean value}}
}
