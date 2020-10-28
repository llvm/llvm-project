// RUN: %clang_cc1 -fsyntax-only -verify %s

// From `test/Sema/typo-correction.c` but for C++ since the behavior varies
// between the two languages.
struct rdar38642201 {
  int fieldName;
};

void rdar38642201_callee(int x, int y);
void rdar38642201_caller() {
  struct rdar38642201 structVar;
  rdar38642201_callee(
      structVar1.fieldName1.member1,  //expected-error{{use of undeclared identifier 'structVar1'}}
      structVar2.fieldName2.member2); //expected-error{{use of undeclared identifier 'structVar2'}}
}

// Similar reproducer.
class A {
public:
  int minut() const = delete;
  int hour() const = delete;

  int longit() const; //expected-note{{'longit' declared here}}
  int latit() const;
};

class B {
public:
  A depar() const { return A(); }
};

int Foo(const B &b) {
  return b.deparT().hours() * 60 + //expected-error{{no member named 'deparT' in 'B'}}
         b.deparT().minutes();     //expected-error{{no member named 'deparT' in 'B'}}
}

int Bar(const B &b) {
  return b.depar().longitude() + //expected-error{{no member named 'longitude' in 'A'; did you mean 'longit'?}}
         b.depar().latitude();   //expected-error{{no member named 'latitude' in 'A'}}
}
