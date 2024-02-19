// RUN: %clang_cc1 -fsyntax-only %s -verify -fobjc-exceptions
// expected-no-diagnostics

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
