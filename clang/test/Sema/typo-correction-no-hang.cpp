// RUN: %clang_cc1 -fsyntax-only -verify %s

// From `test/Sema/typo-correction.c` but for C++ since the behavior varies
// between the two languages.
struct rdar38642201 {
  int fieldName;
};

void rdar38642201_callee(int x, int y);
void rdar38642201_caller() {
  struct rdar38642201 structVar;      //expected-note 2{{'structVar' declared here}}
  rdar38642201_callee(
      structVar1.fieldName1.member1,  //expected-error{{use of undeclared identifier 'structVar1'}} \
                                        expected-error{{no member named 'fieldName1' in 'rdar38642201'}}
      structVar2.fieldName2.member2); //expected-error{{use of undeclared identifier 'structVar2'}} \
                                        expected-error{{no member named 'fieldName2' in 'rdar38642201'}}
}

// Similar reproducer.
class A {
public:
  int minut() const = delete;
  int hour() const = delete;

  int longit() const;
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
  return b.depar().longitude() + //expected-error{{no member named 'longitude' in 'A'}}
         b.depar().latitude();   //expected-error{{no member named 'latitude' in 'A'}}
}
