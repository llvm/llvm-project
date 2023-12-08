// Test this without pch.
// RUN: %clang_cc1 -include %S/functions.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t %S/functions.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s 

int f0(int x0, int y0, ...) { return x0 + y0; }

float *test_f1(int val, double x, double y) {
  if (val > 5)
    return f1(x, y);
  else
    return f1(x); // expected-error{{too few arguments to function call}}
                  // expected-note@functions.h:7{{'f1' declared here}}
}

void test_g0(int *x, float * y) {
  g0(y); // expected-warning{{incompatible pointer types passing 'float *' to parameter of type 'int *'}}
         // expected-note@functions.h:9{{passing argument to parameter here}}
  g0(x); 
}

void __attribute__((noreturn)) test_abort(int code) {
  do_abort(code);
}
  
