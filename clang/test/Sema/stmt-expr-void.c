// RUN: %clang_cc1 -emit-llvm-only -verify %s

void test_benign_null_declgroup(void) {
    // expected-warning@+1 {{declaration does not declare anything}}
    (void)({ extern void; });
}

extern int bar(int); // expected-note {{passing argument to parameter here}}

int foo(int x) {
  // expected-error@+1 {{passing 'void' to parameter of incompatible type 'int'}}
  return 1 + bar(({
    switch (x) { default: break; }
    // expected-warning@+1 {{declaration does not declare anything}}
    extern void;
  }));
}
