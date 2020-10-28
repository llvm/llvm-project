// RUN: %clang_cc1 -fsyntax-only -verify %s

// Check the following typo correction behavior:
// - multiple typos in a single member call chain are all diagnosed
// - no typos are diagnosed for multiple typos in an expression when not all
//   typos can be corrected

class DeepClass
{
public:
  void trigger() const;  // expected-note {{'trigger' declared here}}
};

class Y
{
public:
  const DeepClass& getX() const { return m_deepInstance; }  // expected-note {{'getX' declared here}}
private:
  DeepClass m_deepInstance;
  int m_n;
};

class Z
{
public:
  const Y& getY0() const { return m_y0; }  // expected-note {{'getY0' declared here}}
  const Y& getActiveY() const { return m_y0; }

private:
  Y m_y0;
  Y m_y1;
};

Z z_obj;

void testMultipleCorrections()
{
  z_obj.getY2().  // expected-error {{no member named 'getY2' in 'Z'; did you mean 'getY0'}}
      getM().     // expected-error {{no member named 'getM' in 'Y'; did you mean 'getX'}}
      triggee();  // expected-error {{no member named 'triggee' in 'DeepClass'; did you mean 'trigger'}}
}

void testNoCorrections()
{
  z_obj.getY2().  // expected-error {{no member named 'getY2' in 'Z'}}
      getM().
      thisDoesntSeemToMakeSense();
}

struct C {};
struct D { int value; };
struct A {
  C get_me_a_C();
};
struct B {
  D get_me_a_D();  // expected-note {{'get_me_a_D' declared here}}
};
class Scope {
public:
  A make_an_A();
  B make_a_B();  // expected-note {{'make_a_B' declared here}}
};

Scope scope_obj;

int testDiscardedCorrections() {
  return scope_obj.make_an_E().  // expected-error {{no member named 'make_an_E' in 'Scope'; did you mean 'make_a_B'}}
      get_me_a_Z().value;        // expected-error {{no member named 'get_me_a_Z' in 'B'; did you mean 'get_me_a_D'}}
}

class AmbiguousHelper {
public:
  int helpMe();
  int helpBe();
};
class Ambiguous {
public:
  int calculateA();
  int calculateB();

  AmbiguousHelper getHelp1();
  AmbiguousHelper getHelp2();
};

Ambiguous ambiguous_obj;

int testDirectAmbiguousCorrection() {
  return ambiguous_obj.calculateZ();  // expected-error {{no member named 'calculateZ' in 'Ambiguous'}}
}

int testRecursiveAmbiguousCorrection() {
  return ambiguous_obj.getHelp3().    // expected-error {{no member named 'getHelp3' in 'Ambiguous'}}
      helpCe();
}


class DeepAmbiguityHelper {
public:
  DeepAmbiguityHelper& help1();
  DeepAmbiguityHelper& help2();

  DeepAmbiguityHelper& methodA();
  DeepAmbiguityHelper& somethingMethodB();
  DeepAmbiguityHelper& functionC();
  DeepAmbiguityHelper& deepMethodD();
  DeepAmbiguityHelper& asDeepAsItGets();
};

DeepAmbiguityHelper deep_obj;

int testDeepAmbiguity() {
  deep_obj.
      methodB(). // expected-error {{no member named 'methodB' in 'DeepAmbiguityHelper'}}
      somethingMethodC().
      functionD().
      deepMethodD().
      help3().
      asDeepASItGet().
      functionE();
}

struct Dog {
  int age;  //expected-note{{'age' declared here}}
  int size; //expected-note{{'size' declared here}}
};

int from_dog_years(int DogYears, int DogSize);
int get_dog_years() {
  struct Dog doggo;
  return from_dog_years(doggo.agee,   //expected-error{{no member named 'agee' in 'Dog'; did you mean 'age'}}
                        doggo.sizee); //expected-error{{no member named 'sizee' in 'Dog'; did you mean 'size'}}
}
