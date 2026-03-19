// RUN: %clang_cc1 -Wno-error=return-type -Wno-return-mismatch -fsyntax-only -verify=return-type %s
// RUN: %clang_cc1 -Wno-return-type -Wreturn-mismatch -fsyntax-only -verify=return-mismatch %s

int foo(void) __attribute__((noreturn));
int bar(void);

void test1(void) {
  return 1; // return-mismatch-warning{{void function 'test1' should not return a value}}
}

int test2(void) { 
    return; // return-mismatch-warning{{non-void function 'test2' should return a value}}
} 

int test3(void) { 
    // return-type-warning@+1 {{non-void function does not return a value}}
} 

int test4(void) {
    (void)(bar() || foo()); // return-type-warning@+1 {{non-void function does not return a value in all control paths}}
} 

void test5(void) {
} // no-warning

int test6(void) {
  return 0; // no-warning
}

int test7(void) {
  foo(); // no warning
}

int test8(void) {
  bar(); // return-type-warning@+1 {{non-void function does not return a value}}
}
