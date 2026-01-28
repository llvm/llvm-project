// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.NoDeleteChecker -verify %s

void someFunction();
void [[clang::annotate_type("webkit.nodelete")]] safeFunction();

void [[clang::annotate_type("webkit.nodelete")]] callsUnsafe() {
  // expected-warning@-1{{A function 'callsUnsafe' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  someFunction();
}

void [[clang::annotate_type("webkit.nodelete")]] callsSafe() {
  safeFunction();
}

void [[clang::annotate_type("webkit.nodelete")]] declWithNoDelete();
void declWithNoDelete() {
  // expected-warning@-1{{A function 'declWithNoDelete' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  someFunction();
}

void defWithNoDelete();
void [[clang::annotate_type("webkit.nodelete")]] defWithNoDelete() {
// expected-warning@-1{{A function 'defWithNoDelete' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  someFunction();
}

class SomeClass {
  void [[clang::annotate_type("webkit.nodelete")]] someMethod();
  void [[clang::annotate_type("webkit.nodelete")]] unsafeMethod() {
    // expected-warning@-1{{A function 'unsafeMethod' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
    someFunction();
  }
  void [[clang::annotate_type("webkit.nodelete")]] safeMethod() {
    safeFunction();
  }

  virtual void [[clang::annotate_type("webkit.nodelete")]] someVirtualMethod();
  virtual void [[clang::annotate_type("webkit.nodelete")]] unsafeVirtualMethod() {
    // expected-warning@-1{{A function 'unsafeVirtualMethod' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
    someFunction();
  }
  virtual void [[clang::annotate_type("webkit.nodelete")]] safeVirtualMethod() {
    safeFunction();
  }

  static void [[clang::annotate_type("webkit.nodelete")]] someStaticMethod();
  static void [[clang::annotate_type("webkit.nodelete")]] unsafeStaticMethod() {
    // expected-warning@-1{{A function 'unsafeStaticMethod' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
    someFunction();
  }
  static void [[clang::annotate_type("webkit.nodelete")]] safeStaticMethod() {
    safeFunction();
  }

  virtual void [[clang::annotate_type("webkit.nodelete")]] anotherVirtualMethod();
};

class IntermediateClass : public SomeClass {
  void anotherVirtualMethod() override;
};

class DerivedClass : public IntermediateClass {
  void anotherVirtualMethod() override {
    // expected-warning@-1{{A function 'anotherVirtualMethod' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
    someFunction();
  }
};

template <class Type>
class Base {
public:
  virtual unsigned foo() const = 0;
};

template <class Type>
class Derived : public Base<Type> {
public:
  virtual unsigned foo() const { return 0; }
};
