// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -fobjc-arc -analyzer-ipa=inlining -analyzer-config c++-inlining=constructors -Wno-null-dereference -verify %s

void clang_analyzer_eval(bool);
void clang_analyzer_checkInlined(bool);

struct Wrapper {
  __strong id obj;
};

void test() {
  Wrapper w;
  // force a diagnostic
  *(char *)0 = 1; // expected-warning{{Dereference of null pointer}}
}


struct IntWrapper {
  int x;
};

void testCopyConstructor() {
  IntWrapper a;
  a.x = 42;

  IntWrapper b(a);
  clang_analyzer_eval(b.x == 42); // expected-warning{{TRUE}}
}

struct NonPODIntWrapper {
  int x;

  virtual int get();
};

void testNonPODCopyConstructor() {
  NonPODIntWrapper a;
  a.x = 42;

  NonPODIntWrapper b(a);
  clang_analyzer_eval(b.x == 42); // expected-warning{{TRUE}}
}


namespace ConstructorVirtualCalls {
  class A {
  public:
    int *out1, *out2, *out3;

    virtual int get() { return 1; }

    A(int *out1) {
      *out1 = get();
    }
  };

  class B : public A {
  public:
    virtual int get() { return 2; }

    B(int *out1, int *out2) : A(out1) {
      *out2 = get();
    }
  };

  class C : public B {
  public:
    virtual int get() { return 3; }

    C(int *out1, int *out2, int *out3) : B(out1, out2) {
      *out3 = get();
    }
  };

  void test() {
    int a, b, c;

    C obj(&a, &b, &c);
    clang_analyzer_eval(a == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(b == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(c == 3); // expected-warning{{TRUE}}

    clang_analyzer_eval(obj.get() == 3); // expected-warning{{TRUE}}

    // Sanity check for devirtualization.
    A *base = &obj;
    clang_analyzer_eval(base->get() == 3); // expected-warning{{TRUE}}
  }
}

namespace TemporaryConstructor {
  class BoolWrapper {
  public:
    BoolWrapper() {
      clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
      value = true;
    }
    bool value;
  };

  void test() {
    // PR13717 - Don't crash when a CXXTemporaryObjectExpr is inlined.
    if (BoolWrapper().value)
      return;
  }
}


namespace ConstructorUsedAsRValue {
  using TemporaryConstructor::BoolWrapper;

  bool extractValue(BoolWrapper b) {
    return b.value;
  }

  void test() {
    bool result = extractValue(BoolWrapper());
    clang_analyzer_eval(result); // expected-warning{{TRUE}}
  }
}
