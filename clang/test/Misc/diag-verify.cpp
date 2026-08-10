// RUN: not %clang_cc1 -fsyntax-only -verify %s 2>&1 | FileCheck %s

// Test the -verify flag.  Each of the "x = y;" lines will produce a
// "use of undeclared identifier 'y'" error message.

void test() {
  int x;
  // Proper matches here.
  x = y; // expected-error{{use of undeclared identifier 'y'}}
  x = y; // expected-error{{use of undeclared identifier}}
  x = y; // expected-error{{undeclared identifier 'y'}}
  x = y; // expected-error{{use of}}
  x = y; // expected-error{{undeclared identifier}}
  x = y; // expected-error{{'y'}}

  // Bad matches here.
  x = y; // expected-error{{use of undeclared identifier 'y' is fine}}
  x = y; // expected-error{{abuse of undeclared identifier 'y'}}
  x = y; // expected-error{{good use of undeclared identifier 'y' in code}}
  x = y; // expected-error{{ use of undeclared identifier 'y' }}
  x = y; // expected-error{{use of undeclared identifier 'y' is disallowed}}
  x = y; // expected-error{{please don't use of undeclared identifier 'y'}}
  x = y; // expected-error{{use of undeclared identifier 'y'; please declare y before use}}
  x = y; // expected-error{{use of use of undeclared identifier 'y'}}
  x = y; // expected-error{{use of undeclared identifier 'y' identifier 'y'}}
}

//CHECK: error: 'expected-error' diagnostics expected but not seen: 
//CHECK:   Line 17: use of undeclared identifier 'y' is fine
//CHECK:   Line 18: abuse of undeclared identifier 'y'
//CHECK:   Line 19: good use of undeclared identifier 'y' in code
//CHECK:   Line 20:  use of undeclared identifier 'y' 
//CHECK:   Line 21: use of undeclared identifier 'y' is disallowed
//CHECK:   Line 22: please don't use of undeclared identifier 'y'
//CHECK:   Line 23: use of undeclared identifier 'y'; please declare y before use
//CHECK:   Line 24: use of use of undeclared identifier 'y'
//CHECK:   Line 25: use of undeclared identifier 'y' identifier 'y'
//CHECK: error: 'expected-error' diagnostics seen but not expected: 
//CHECK:   Line 17: use of undeclared identifier 'y'
//CHECK:   Line 18: use of undeclared identifier 'y'
//CHECK:   Line 19: use of undeclared identifier 'y'
//CHECK:   Line 20: use of undeclared identifier 'y'
//CHECK:   Line 21: use of undeclared identifier 'y'
//CHECK:   Line 22: use of undeclared identifier 'y'
//CHECK:   Line 23: use of undeclared identifier 'y'
//CHECK:   Line 24: use of undeclared identifier 'y'
//CHECK:   Line 25: use of undeclared identifier 'y'
//CHECK: 18 errors generated.
