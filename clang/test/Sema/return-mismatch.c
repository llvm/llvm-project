// RUN: %clang_cc1 -Wno-return-type -Wreturn-mismatch -fsyntax-only -verify %s

// Test that -Wreturn-mismatch is enabled and -Wreturn-type is disabled.

int foo(void) __attribute__((noreturn));
int bar(void);

void test1() {
  return 1; // expected-warning{{void function 'test1' should not return a value}}
}

int test2() { 
    return; // expected-warning{{non-void function 'test2' should return a value}}
} 

int test3() { 
    // no-warning
} 

int test4() {
    (void)(bar() || foo()); // no-warning
} 

void test5() {
} // no-warning

int test6() {
  return 0; // no-warning
}