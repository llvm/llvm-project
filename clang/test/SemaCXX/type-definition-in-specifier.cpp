// RUN: %clang_cc1 -fexceptions -fcxx-exceptions -verify %s

struct S0;
struct S1;
struct S2;
struct S3;
struct S4;
struct S5;
struct S6;

struct S0 { int x; };

void f0() {
  typedef struct S1 { int x; } S1_typedef;

  (void)((struct S2 { int x; }*)0); // expected-error{{cannot be defined}}

  struct S3 { int x; } s3;

  (void)static_cast<struct S4 { int x; } *>(0); // expected-error{{cannot be defined}}
}

struct S5 { int x; } f1() { return S5(); } // expected-error{{result type}}

void f2(struct S6 { int x; } p); // expected-error{{parameter type}}

struct pr19018 {
  short foo6 (enum bar0 {qq} bar3); // expected-error{{cannot be defined in a parameter type}}
};

void pr19018_1 (enum e19018_1 {qq} x); // expected-error{{cannot be defined in a parameter type}}
void pr19018_1a (enum e19018_1 {qq} x); // expected-error{{cannot be defined in a parameter type}}
e19018_1 x2;

void pr19018_2 (enum {qq} x); // expected-error{{cannot be defined in a parameter type}}
void pr19018_3 (struct s19018_2 {int qq;} x); // expected-error{{cannot be defined in a parameter type}}
void pr19018_4 (struct {int qq;} x); // expected-error{{cannot be defined in a parameter type}}
void pr19018_5 (struct { void qq(); } x); // expected-error{{cannot be defined in a parameter type}}
void pr19018_5 (struct s19018_2 { void qq(); } x); // expected-error{{cannot be defined in a parameter type}}

struct pr19018a {
  static int xx;
  void func1(enum t19018 {qq} x); // expected-error{{cannot be defined in a parameter type}}
  void func2(enum t19018 {qq} x); // expected-error{{cannot be defined in a parameter type}}
  void func3(enum {qq} x);        // expected-error{{cannot be defined in a parameter type}}
  void func4(struct t19018 {int qq;} x);  // expected-error{{cannot be defined in a parameter type}}
  void func5(struct {int qq;} x); // expected-error{{cannot be defined in a parameter type}}
  void func6(struct { void qq(); } x); // expected-error{{cannot be defined in a parameter type}}
  void func7(struct t19018 { void qq(); } x); // expected-error{{cannot be defined in a parameter type}}
  void func8(struct { int qq() { return xx; }; } x); // expected-error{{cannot be defined in a parameter type}}
  void func9(struct t19018 { int qq() { return xx; }; } x); // expected-error{{cannot be defined in a parameter type}}
};

struct s19018b {
  void func1 (enum en_2 {qq} x); // expected-error{{cannot be defined in a parameter type}}
  en_2 x1;
  void func2 (enum en_3 {qq} x); // expected-error{{cannot be defined in a parameter type}}
  enum en_3 x2;
};

struct pr18963 {
  short bar5 (struct foo4 {} bar2); // expected-error{{'foo4' cannot be defined in a parameter type}}
  long foo5 (float foo6 = foo4);
};

// expected-error@+2 {{cannot be defined in a parameter type}}
// expected-note@+1 {{previous definition is here}}
void func_with_eh_and_type(struct type_in_eh {} o) throw(int) {}
struct type_in_eh {}; // expected-error {{redefinition of 'type_in_eh'}}
