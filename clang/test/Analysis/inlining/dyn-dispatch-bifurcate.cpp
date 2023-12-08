// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config ipa=dynamic-bifurcate -verify -Wno-reinterpret-base-class -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(bool);

class A {
public:
  virtual int get() { return 0; }
};

void testBifurcation(A *a) {
  clang_analyzer_eval(a->get() == 0); // expected-warning{{TRUE}} expected-warning{{UNKNOWN}}
}

void testKnown() {
  A a;
  clang_analyzer_eval(a.get() == 0); // expected-warning{{TRUE}}
}

void testNew() {
  A *a = new A();
  clang_analyzer_eval(a->get() == 0); // expected-warning{{TRUE}}
}


namespace ReinterpretDisruptsDynamicTypeInfo {
  class Parent {};

  class Child : public Parent {
  public:
    virtual int foo() { return 42; }
  };

  void test(Parent *a) {
    Child *b = reinterpret_cast<Child *>(a);
    if (!b) return;
    clang_analyzer_eval(b->foo() == 42); // expected-warning{{UNKNOWN}}
  }
}
