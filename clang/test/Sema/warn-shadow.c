// RUN: %clang_cc1 -verify -fsyntax-only -fblocks -Wshadow %s

int i;          // expected-note 4 {{previous declaration is here}}
static int s;   // expected-note 2 {{previous declaration is here}}

void foo(void) {
  int pass1;
  int i;        // expected-warning {{declaration shadows a variable in the global scope}} \
                // expected-note {{previous declaration is here}}
  int s;        // expected-warning {{declaration shadows a variable in the global scope}} \
                // expected-note {{previous declaration is here}}
  {
    int pass2;
    int i;      // expected-warning {{declaration shadows a local variable}} \
                // expected-note {{previous declaration is here}}
    int s;      // expected-warning {{declaration shadows a local variable}} \
                // expected-note {{previous declaration is here}}
    {
      int pass3;
      int i;    // expected-warning {{declaration shadows a local variable}}
      int s;    // expected-warning {{declaration shadows a local variable}}
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
void rdar8883302(void) {
  extern int bob; // don't warn for shadowing.
}

void test8(void) {
  int bob; // expected-warning {{declaration shadows a variable in the global scope}}
}

enum PR24718_1{pr24718}; // expected-note {{previous declaration is here}}
void PR24718(void) {
  enum PR24718_2{pr24718}; // expected-warning {{declaration shadows a variable in the global scope}}
}

struct PR24718_3;
struct PR24718_4 {
  enum {
    PR24718_3 // Does not shadow a type.
  };
};

void static_locals() {
  static int i; // expected-warning {{declaration shadows a variable in the global scope}}
                // expected-note@-1 {{previous definition is here}}
                // expected-note@-2 {{previous declaration is here}}
  int i;        // expected-error {{non-static declaration of 'i' follows static declaration}}
  static int foo; // expected-note {{previous declaration is here}}
  static int hoge; // expected-note {{previous declaration is here}}
  int s; // expected-warning {{declaration shadows a variable in the global scope}}
  {
    static int foo; // expected-warning {{declaration shadows a local variable}}
                    // expected-note@-1 {{previous declaration is here}}
    static int i; // expected-warning {{declaration shadows a local variable}}
                  // expected-note@-1 {{previous declaration is here}}
    int hoge; // expected-warning {{declaration shadows a local variable}}
    {
      static int foo; // expected-warning {{declaration shadows a local variable}}
      int i; // expected-warning {{declaration shadows a local variable}}
      extern int hoge;
    }
  }
}
