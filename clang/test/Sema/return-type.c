// RUN: %clang_cc1 -Wreturn-type -Wno-return-mismatch -fsyntax-only -verify %s

// Test that -Wreturn-type is enabled and -Wreturn-mismatch is disabled.

int foo(void) __attribute__((noreturn));
int bar(void);

void test1() {
  return 1; // no-warning
}

int test2() { 
    return; // no-warning
} 

int test3() { 
    // expected-warning@+1 {{non-void function does not return a value}}
} 

int test4() {
    (void)(bar() || foo()); // expected-warning@+1 {{non-void function does not return a value in all control paths}}
} 

void test5() {
} // no-warning

int test6() {
  return 0; // no-warning
}