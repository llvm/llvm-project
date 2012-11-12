// RUN: %clang_cc1 -verify -fsyntax-only -fblocks -Wshadow %s

int i;          // expected-note 3 {{previous declaration is here}}

void foo() {
  int pass1;
  int i;        // expected-warning {{declaration shadows a variable in the global scope}} \
                // expected-note {{previous declaration is here}}
  {
    int pass2;
    int i;      // expected-warning {{declaration shadows a local variable}} \
                // expected-note {{previous declaration is here}}
    {
      int pass3;
      int i;    // expected-warning {{declaration shadows a local variable}}
    }
  }

  int sin; // okay; 'sin' has not been declared, even though it's a builtin.
}

// <rdar://problem/7677531>
void (^test1)(int) = ^(int i) { // expected-warning {{declaration shadows a variable in the global scope}} \
                                 // expected-note{{previous declaration is here}}
  {
    int i; // expected-warning {{declaration shadows a local variable}} \
           // expected-note{{previous declaration is here}}
    
    (^(int i) { return i; })(i); //expected-warning {{declaration shadows a local variable}}
  }
};


struct test2 {
  int i;
};

void test3(void) {
  struct test4 {
    int i;
  };
}

void test4(int i) { // expected-warning {{declaration shadows a variable in the global scope}}
}

// Don't warn about shadowing for function declarations.
void test5(int i);
void test6(void (*f)(int i)) {}
void test7(void *context, void (*callback)(void *context)) {}

extern int bob; // expected-note {{previous declaration is here}}

// rdar://8883302
void rdar8883302() {
  extern int bob; // don't warn for shadowing.
}

void test8() {
  int bob; // expected-warning {{declaration shadows a variable in the global scope}}
}
