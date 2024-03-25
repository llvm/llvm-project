// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=core,optin.core.EnumCastOutOfRange \
// RUN:   -analyzer-output text \
// RUN:   -verify %s

// expected-note@+1 + {{enum declared here}}
enum En_t {
  En_0 = -4,
  En_1,
  En_2 = 1,
  En_3,
  En_4 = 4
};

void unscopedUnspecifiedCStyle(void) {
  enum En_t Below = (enum En_t)(-5);    // expected-warning {{not in the valid range of values for 'En_t'}}
                                        // expected-note@-1 {{not in the valid range of values for 'En_t'}}
  enum En_t NegVal1 = (enum En_t)(-4);  // OK.
  enum En_t NegVal2 = (enum En_t)(-3);  // OK.
  enum En_t InRange1 = (enum En_t)(-2); // expected-warning {{not in the valid range of values for 'En_t'}}
                                        // expected-note@-1 {{not in the valid range of values for 'En_t'}}
  enum En_t InRange2 = (enum En_t)(-1); // expected-warning {{not in the valid range of values for 'En_t'}}
                                        // expected-note@-1 {{not in the valid range of values for 'En_t'}}
  enum En_t InRange3 = (enum En_t)(0);  // expected-warning {{not in the valid range of values for 'En_t'}}
                                        // expected-note@-1 {{not in the valid range of values for 'En_t'}}
  enum En_t PosVal1 = (enum En_t)(1);   // OK.
  enum En_t PosVal2 = (enum En_t)(2);   // OK.
  enum En_t InRange4 = (enum En_t)(3);  // expected-warning {{not in the valid range of values for 'En_t'}}
                                        // expected-note@-1 {{not in the valid range of values for 'En_t'}}
  enum En_t PosVal3 = (enum En_t)(4);   // OK.
  enum En_t Above = (enum En_t)(5);     // expected-warning {{not in the valid range of values for 'En_t'}}
                                        // expected-note@-1 {{not in the valid range of values for 'En_t'}}
}

enum En_t unused;
void unusedExpr(void) {
  // Following line is not something that EnumCastOutOfRangeChecker should
  // evaluate.  Checker should either ignore this line or process it without
  // producing any warnings.  However, compilation will (and should) still
  // generate a warning having nothing to do with this checker.
  unused; // expected-warning {{expression result unused}}
}

// Test typedef-ed anonymous enums
typedef enum { // expected-note {{enum declared here}}
    TD_0 = 0,
} TD_t;

void testTypeDefEnum(void) {
  (void)(TD_t)(-1); // expected-warning {{not in the valid range of values for the enum}}
                    // expected-note@-1 {{not in the valid range of values for the enum}}
}

// Test expression tracking
void set(int* p, int v) {
  *p = v; // expected-note {{The value -1 is assigned to 'i'}}
}


void testTrackExpression(int i) {
  set(&i, -1); // expected-note {{Passing the value -1 via 2nd parameter 'v'}}
               // expected-note@-1 {{Calling 'set'}}
               // expected-note@-2 {{Returning from 'set'}}
  (void)(enum En_t)(i); // expected-warning {{not in the valid range of values for 'En_t'}}
                        // expected-note@-1 {{not in the valid range of values for 'En_t'}}
}
