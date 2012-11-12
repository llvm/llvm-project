// RUN: %clang_cc1 -fsyntax-only -verify %s

void f() const; // expected-error {{non-member function cannot have 'const' qualifier}}
void (*pf)() const; // expected-error {{pointer to function type cannot have 'const' qualifier}}
extern void (&rf)() const; // expected-error {{reference to function type cannot have 'const' qualifier}}

typedef void cfn() const;
cfn f2; // expected-error {{non-member function of type 'cfn' (aka 'void () const') cannot have 'const' qualifier}}

class C {
  void f() const;
  cfn f2;
  static void f3() const; // expected-error {{static member function cannot have 'const' qualifier}}
  static cfn f4; // expected-error {{static member function of type 'cfn' (aka 'void () const') cannot have 'const' qualifier}}

  void m1() {
    x = 0;
  }

  void m2() const {
    x = 0; // expected-error {{read-only variable is not assignable}}
  }

  int x;
};

void (C::*mpf)() const;
cfn C::*mpg;

// Don't crash!
void (PR14171)() const; // expected-error {{non-member function cannot have 'const' qualifier}}
