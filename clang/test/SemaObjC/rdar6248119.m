// RUN: %clang_cc1 -fsyntax-only %s -verify -fobjc-exceptions
// expected-no-diagnostics
// Test case for: 
//   <rdar://problem/6248119> @finally doesn't introduce a new scope

void f0(void) {
  int i;
  @try { 
  } @finally {
    int i = 0;
  }
}

void f1(void) {
  int i;
  @try { 
    int i =0;
  } @finally {
  }
}

void f2(void) {
  int i;
  @try { 
  } @catch(id e) {
    int i = 0;
  }
}
