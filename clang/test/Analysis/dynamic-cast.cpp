// RUN: %clang_cc1 -triple i386-apple-darwin10 -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-ipa=none -verify %s

void clang_analyzer_eval(bool);

class A {
public:
    virtual void f(){};

};
class B : public A{
public:
  int m;
};
class C : public A{};

class BB: public B{};

// A lot of the tests below have the if statement in them, which forces the
// analyzer to explore both path - when the result is 0 and not. This makes
// sure that we definitely know that the result is non-0 (as the result of
// the cast).
int testDynCastFromRadar() {
    B aa;
    A *a = &aa;
    const int* res = 0;
    B *b = dynamic_cast<B*>(a);
    static const int i = 5;
    if(b) {
        res = &i;
    } else {
        res = 0;
    }
    return *res; // no warning
}

int testBaseToBase1() {
  B b;
  B *pb = &b;
  B *pbb = dynamic_cast<B*>(pb);
  const int* res = 0;
  static const int i = 5;
  if (pbb) {
      res = &i;
  } else {
      res = 0;
  }
  return *res; // no warning
}

int testMultipleLevelsOfSubclassing1() {
  BB bb;
  B *pb = &bb;
  A *pa = pb;
  B *b = dynamic_cast<B*>(pa);
  const int* res = 0;
  static const int i = 5;
  if (b) {
      res = &i;
  } else {
      res = 0;
  }
  return *res; // no warning
}

int testMultipleLevelsOfSubclassing2() {
  BB bb;
  A *pbb = &bb;
  B *b = dynamic_cast<B*>(pbb);
  BB *s = dynamic_cast<BB*>(b);
  const int* res = 0;
  static const int i = 5;
  if (s) {
      res = &i;
  } else {
      res = 0;
  }
  return *res; // no warning
}

int testMultipleLevelsOfSubclassing3() {
  BB bb;
  A *pbb = &bb;
  B *b = dynamic_cast<B*>(pbb);
  return b->m; // no warning
}

int testLHS() {
    B aa;
    A *a = &aa;
    return (dynamic_cast<B*>(a))->m;
}

int testLHS2() {
    B aa;
    A *a = &aa;
    return (*dynamic_cast<B*>(a)).m;
}

int testDynCastUnknown2(class A *a) {
  B *b = dynamic_cast<B*>(a);
  return b->m; // no warning
}

int testDynCastUnknown(class A *a) {
  B *b = dynamic_cast<B*>(a);
  const int* res = 0;
  static const int i = 5;
  if (b) {
    res = &i;
  } else {
    res = 0;
  }
  return *res; // expected-warning {{Dereference of null pointer}}
}

int testDynCastFail2() {
  C c;
  A *pa = &c;
  B *b = dynamic_cast<B*>(pa);
  return b->m; // expected-warning {{dereference of a null pointer}}
}

int testLHSFail() {
    C c;
    A *a = &c;
    return (*dynamic_cast<B*>(a)).m; // expected-warning {{Dereference of null pointer}}
}

int testBaseToDerivedFail() {
  A a;
  B *b = dynamic_cast<B*>(&a);
  return b->m; // expected-warning {{dereference of a null pointer}}
}

int testConstZeroFail() {
  B *b = dynamic_cast<B*>((A *)0);
  return b->m; // expected-warning {{dereference of a null pointer}}
}

int testConstZeroFail2() {
  A *a = 0;
  B *b = dynamic_cast<B*>(a);
  return b->m; // expected-warning {{dereference of a null pointer}}
}

int testUpcast() {
  B b;
  A *a = dynamic_cast<A*>(&b);
  const int* res = 0;
  static const int i = 5;
  if (a) {
      res = &i;
  } else {
      res = 0;
  }
  return *res; // no warning
}

int testCastToVoidStar() {
  A a;
  void *b = dynamic_cast<void*>(&a);
  const int* res = 0;
  static const int i = 5;
  if (b) {
      res = &i;
  } else {
      res = 0;
  }
  return *res; // no warning
}

int testReferenceSuccesfulCast() {
  B rb;
  B &b = dynamic_cast<B&>(rb);
  int *x = 0;
  return *x; // expected-warning {{Dereference of null pointer}}
}

int testReferenceFailedCast() {
  A a;
  B &b = dynamic_cast<B&>(a);
  int *x = 0;
  return *x; // no warning (An exception is thrown by the cast.)
}

// Here we allow any outcome of the cast and this is good because there is a
// situation where this will fail. So if the user has written the code in this
// way, we assume they expect the cast to succeed.
// Note, this might need special handling if we track types of symbolic casts
// and use them for dynamic_cast handling.
int testDynCastMostLikelyWillFail(C *c) {
  B *b = 0;
  b = dynamic_cast<B*>(c);
  const int* res = 0;
  static const int i = 5;
  if (b) {
      res = &i;
  } else {
      res = 0;
  }

  // Note: IPA is turned off for this test because the code below shows how the
  // dynamic_cast could succeed.
  return *res; // expected-warning{{Dereference of null pointer}}
}

class M : public B, public C {};
void callTestDynCastMostLikelyWillFail() {
  M m;
  testDynCastMostLikelyWillFail(&m);
}


void testDynCastToMiddleClass () {
  class BBB : public BB {};
  BBB obj;
  A &ref = obj;

  // These didn't always correctly layer base regions.
  B *ptr = dynamic_cast<B*>(&ref);
  clang_analyzer_eval(ptr != 0); // expected-warning{{TRUE}}

  // This is actually statically resolved to be a DerivedToBase cast.
  ptr = dynamic_cast<B*>(&obj);
  clang_analyzer_eval(ptr != 0); // expected-warning{{TRUE}}
}


// -----------------------------
// False positives/negatives.
// -----------------------------

// Due to symbolic regions not being typed.
int testDynCastFalsePositive(BB *c) {
  B *b = 0;
  b = dynamic_cast<B*>(c);
  const int* res = 0;
  static const int i = 5;
  if (b) {
      res = &i;
  } else {
      res = 0;
  }
  return *res; // expected-warning{{Dereference of null pointer}}
}

// Does not work when we new an object.
int testDynCastFail3() {
  A *a = new A();
  B *b = dynamic_cast<B*>(a);
  return b->m;
}

