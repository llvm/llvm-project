// RUN: %clang_analyze_cc1 -Wno-error=return-type -analyzer-checker=core,debug.ExprInspection -verify -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(bool);

struct A {
  int x;
  void foo() const;
  void bar();

  void testImplicitThisSyntax() {
    x = 3;
    foo();
    clang_analyzer_eval(x == 3); // expected-warning{{TRUE}}
    bar();
    clang_analyzer_eval(x == 3); // expected-warning{{UNKNOWN}}
  }
};

struct B {
  mutable int mut;
  void foo() const;
};

struct C {
  int *p;
  void foo() const;
};

struct MutBase {
  mutable int b_mut;
};

struct MutDerived : MutBase {
  void foo() const;
};

struct PBase {
  int *p;
};

struct PDerived : PBase {
  void foo() const;
};

struct Inner {
  int x;
  int *p;
  void bar() const;
};

struct Outer {
  int x;
  Inner in;
  void foo() const;
};

void checkThatConstMethodWithoutDefinitionDoesNotInvalidateObject() {
  A t;
  t.x = 3;
  t.foo();
  clang_analyzer_eval(t.x == 3); // expected-warning{{TRUE}}
  // Test non-const does invalidate
  t.bar();
  clang_analyzer_eval(t.x); // expected-warning{{UNKNOWN}}
}

void checkThatConstMethodDoesInvalidateMutableFields() {
  B t;
  t.mut = 4;
  t.foo();
  clang_analyzer_eval(t.mut); // expected-warning{{UNKNOWN}}
}

void checkThatConstMethodDoesInvalidatePointedAtMemory() {
  int x = 1;
  C t;
  t.p = &x;
  t.foo();
  clang_analyzer_eval(x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(t.p == &x); // expected-warning{{TRUE}}
}

void checkThatConstMethodDoesInvalidateInheritedMutableFields() {
  MutDerived t;
  t.b_mut = 4;
  t.foo();
  clang_analyzer_eval(t.b_mut); // expected-warning{{UNKNOWN}}
}

void checkThatConstMethodDoesInvalidateInheritedPointedAtMemory() {
  int x = 1;
  PDerived t;
  t.p = &x;
  t.foo();
  clang_analyzer_eval(x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(t.p == &x); // expected-warning{{TRUE}}
}

void checkThatConstMethodDoesInvalidateContainedPointedAtMemory() {
  int x = 1;
  Outer t;
  t.x = 2;
  t.in.p = &x;
  t.foo();
  clang_analyzer_eval(x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(t.x == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(t.in.p == &x); // expected-warning{{TRUE}}
}

void checkThatContainedConstMethodDoesNotInvalidateObjects() {
  Outer t;
  t.x = 1;
  t.in.x = 2;
  t.in.bar();
  clang_analyzer_eval(t.x == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(t.in.x == 2); // expected-warning{{TRUE}}
}

void checkPointerTypedThisExpression(A *a) {
  a->x = 3;
  a->foo();
  clang_analyzer_eval(a->x == 3); // expected-warning{{TRUE}}
  a->bar();
  clang_analyzer_eval(a->x == 3); // expected-warning{{UNKNOWN}}
}

void checkReferenceTypedThisExpression(A &a) {
  a.x = 3;
  a.foo();
  clang_analyzer_eval(a.x == 3); // expected-warning{{TRUE}}
  a.bar();
  clang_analyzer_eval(a.x == 3); // expected-warning{{UNKNOWN}}
}

// --- Versions of the above tests where the const method is inherited --- //

struct B1 {
  void foo() const;
};

struct D1 : public B1 {
  int x;
};

struct D2 : public B1 {
  mutable int mut;
};

struct D3 : public B1 {
  int *p;
};

struct DInner : public B1 {
  int x;
  int *p;
};

struct DOuter : public B1 {
  int x;
  DInner in;
};

void checkThatInheritedConstMethodDoesNotInvalidateObject() {
  D1 t;
  t.x = 1;
  t.foo();
  clang_analyzer_eval(t.x == 1); // expected-warning{{TRUE}}
}

void checkThatInheritedConstMethodDoesInvalidateMutableFields() {
  D2 t;
  t.mut = 1;
  t.foo();
  clang_analyzer_eval(t.mut); // expected-warning{{UNKNOWN}}
}

void checkThatInheritedConstMethodDoesInvalidatePointedAtMemory() {
  int x = 1;
  D3 t;
  t.p = &x;
  t.foo();
  clang_analyzer_eval(x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(t.p == &x); // expected-warning{{TRUE}}
}

void checkThatInheritedConstMethodDoesInvalidateContainedPointedAtMemory() {
  int x = 1;
  DOuter t;
  t.x = 2;
  t.in.x = 3;
  t.in.p = &x;
  t.foo();
  clang_analyzer_eval(x); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(t.x == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(t.in.x == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(t.in.p == &x); // expected-warning{{TRUE}}
}

void checkThatInheritedContainedConstMethodDoesNotInvalidateObjects() {
  DOuter t;
  t.x = 1;
  t.in.x = 2;
  t.in.foo();
  clang_analyzer_eval(t.x == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(t.in.x == 2); // expected-warning{{TRUE}}
}

// --- PR21606 --- //

struct s1 {
    void g(const int *i) const;
};

struct s2 {
    void f(int *i) {
        m_i = i;
        m_s.g(m_i);
        if (m_i)
            *i = 42; // no-warning
    }

    int *m_i;
    s1 m_s;
};

void PR21606()
{
    s2().f(0);
}

// --- PR25392 --- //

struct HasConstMemberFunction {
public:
  void constMemberFunction() const;
};

HasConstMemberFunction hasNoReturn() { } // expected-warning {{non-void function does not return a value}}

void testUnknownWithConstMemberFunction() {
  hasNoReturn().constMemberFunction();
}

void testNonRegionLocWithConstMemberFunction() {
  (*((HasConstMemberFunction *)(&&label))).constMemberFunction();

  label: return;
}

// FIXME
// When there is a circular reference to an object and a const method is called
// the object is not invalidated because TK_PreserveContents has already been
// set.
struct Outer2;

struct InnerWithRef {
  Outer2 *ref;
};

struct Outer2 {
  int x;
  InnerWithRef in;
  void foo() const;
};

void checkThatConstMethodCallDoesInvalidateObjectForCircularReferences() {
  Outer2 t;
  t.x = 1;
  t.in.ref = &t;
  t.foo();
  // FIXME: Should be UNKNOWN.
  clang_analyzer_eval(t.x); // expected-warning{{TRUE}}
}

namespace gh77378 {
template <typename Signature> class callable;

template <typename R> class callable<R()> {
  struct CallableType {
    bool operator()();
  };
  using MethodType = R (CallableType::*)();
  CallableType *object_{nullptr};
  MethodType method_;

public:
  callable() = default;

  template <typename T>
  constexpr callable(const T &obj)
      : object_{reinterpret_cast<CallableType *>(&const_cast<T &>(obj))},
        method_{reinterpret_cast<MethodType>(
            static_cast<bool (T::*)() const>(&T::operator()))} {}

  constexpr bool const_method() const {
    return (object_->*(method_))();
  }

  callable call() const & {
    static const auto L = [this]() {
      while (true) {
        // This should not crash when conservative eval calling the member function
        // when it unwinds the call stack due to exhausting the budget or reaching
        // the inlining limit.
        if (this->const_method()) {
          break;
        }
      }
      return true;
    };
    return L;
  }
};

void entry() {
  callable<bool()>{}.call().const_method();
  // expected-warning@-1 {{Address of stack memory associated with temporary object of type 'callable<bool ()>' is still referred to by the static variable 'L' upon returning to the caller.  This will be a dangling reference}}
}
} // namespace gh77378
