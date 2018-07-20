// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config cfg-temporary-dtors=false -verify -w -std=c++03 %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config cfg-temporary-dtors=false -verify -w -std=c++11 %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -DTEMPORARY_DTORS -verify -w -analyzer-config cfg-temporary-dtors=true,c++-temp-dtor-inlining=true %s -std=c++11
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -DTEMPORARY_DTORS -w -analyzer-config cfg-temporary-dtors=true,c++-temp-dtor-inlining=true %s -std=c++17

// Note: The C++17 run-line doesn't -verify yet - it is a no-crash test.

extern bool clang_analyzer_eval(bool);
extern bool clang_analyzer_warnIfReached();
void clang_analyzer_checkInlined(bool);

#include "Inputs/system-header-simulator-cxx.h";

struct Trivial {
  Trivial(int x) : value(x) {}
  int value;
};

struct NonTrivial : public Trivial {
  NonTrivial(int x) : Trivial(x) {}
  ~NonTrivial();
};


Trivial getTrivial() {
  return Trivial(42); // no-warning
}

const Trivial &getTrivialRef() {
  return Trivial(42); // expected-warning {{Address of stack memory associated with temporary object of type 'Trivial' returned to caller}}
}


NonTrivial getNonTrivial() {
  return NonTrivial(42); // no-warning
}

const NonTrivial &getNonTrivialRef() {
  return NonTrivial(42); // expected-warning {{Address of stack memory associated with temporary object of type 'NonTrivial' returned to caller}}
}

namespace rdar13265460 {
  struct TrivialSubclass : public Trivial {
    TrivialSubclass(int x) : Trivial(x), anotherValue(-x) {}
    int anotherValue;
  };

  TrivialSubclass getTrivialSub() {
    TrivialSubclass obj(1);
    obj.value = 42;
    obj.anotherValue = -42;
    return obj;
  }

  void testImmediate() {
    TrivialSubclass obj = getTrivialSub();

    clang_analyzer_eval(obj.value == 42); // expected-warning{{TRUE}}
    clang_analyzer_eval(obj.anotherValue == -42); // expected-warning{{TRUE}}

    clang_analyzer_eval(getTrivialSub().value == 42); // expected-warning{{TRUE}}
    clang_analyzer_eval(getTrivialSub().anotherValue == -42); // expected-warning{{TRUE}}
  }

  void testMaterializeTemporaryExpr() {
    const TrivialSubclass &ref = getTrivialSub();
    clang_analyzer_eval(ref.value == 42); // expected-warning{{TRUE}}

    const Trivial &baseRef = getTrivialSub();
    clang_analyzer_eval(baseRef.value == 42); // expected-warning{{TRUE}}
  }
}

namespace rdar13281951 {
  struct Derived : public Trivial {
    Derived(int value) : Trivial(value), value2(-value) {}
    int value2;
  };

  void test() {
    Derived obj(1);
    obj.value = 42;
    const Trivial * const &pointerRef = &obj;
    clang_analyzer_eval(pointerRef->value == 42); // expected-warning{{TRUE}}
  }
}

namespace compound_literals {
  struct POD {
    int x, y;
  };
  struct HasCtor {
    HasCtor(int x, int y) : x(x), y(y) {}
    int x, y;
  };
  struct HasDtor {
    int x, y;
    ~HasDtor();
  };
  struct HasCtorDtor {
    HasCtorDtor(int x, int y) : x(x), y(y) {}
    ~HasCtorDtor();
    int x, y;
  };

  void test() {
    clang_analyzer_eval(((POD){1, 42}).y == 42); // expected-warning{{TRUE}}
    clang_analyzer_eval(((HasDtor){1, 42}).y == 42); // expected-warning{{TRUE}}

#if __cplusplus >= 201103L
    clang_analyzer_eval(((HasCtor){1, 42}).y == 42); // expected-warning{{TRUE}}

    // FIXME: should be TRUE, but we don't inline the constructors of
    // temporaries because we can't model their destructors yet.
    clang_analyzer_eval(((HasCtorDtor){1, 42}).y == 42); // expected-warning{{UNKNOWN}}
#endif
  }
}

namespace destructors {
  struct Dtor {
    ~Dtor();
  };
  extern bool coin();
  extern bool check(const Dtor &);

  void testPR16664andPR18159Crash() {
    // Regression test: we used to assert here when tmp dtors are enabled.
    // PR16664 and PR18159
    if (coin() && (coin() || coin() || check(Dtor()))) {
      Dtor();
    }
  }

#ifdef TEMPORARY_DTORS
  struct NoReturnDtor {
    ~NoReturnDtor() __attribute__((noreturn));
  };

  void noReturnTemp(int *x) {
    if (! x) NoReturnDtor();
    *x = 47; // no warning
  }

  void noReturnInline(int **x) {
    NoReturnDtor();
  }

  void callNoReturn() {
    int *x;
    noReturnInline(&x);
    *x = 47; // no warning
  }

  extern bool check(const NoReturnDtor &);

  void testConsistencyIf(int i) {
    if (i != 5)
      return;
    if (i == 5 && (i == 4 || check(NoReturnDtor()) || i == 5)) {
      clang_analyzer_eval(true); // no warning, unreachable code
    }
  }

  void testConsistencyTernary(int i) {
    (i == 5 && (i == 4 || check(NoReturnDtor()) || i == 5)) ? 1 : 0;

    clang_analyzer_eval(true);  // expected-warning{{TRUE}}

    if (i != 5)
      return;

    (i == 5 && (i == 4 || check(NoReturnDtor()) || i == 5)) ? 1 : 0;

    clang_analyzer_eval(true); // no warning, unreachable code
  }

  // Regression test: we used to assert here.
  // PR16664 and PR18159
  void testConsistencyNested(int i) {
    extern bool compute(bool);

    if (i == 5 && (i == 4 || i == 5 || check(NoReturnDtor())))
      clang_analyzer_eval(true);  // expected-warning{{TRUE}}

    if (i == 5 && (i == 4 || i == 5 || check(NoReturnDtor())))
      clang_analyzer_eval(true);  // expected-warning{{TRUE}}

    if (i != 5)
      return;

    if (compute(i == 5 &&
                (i == 4 || compute(true) ||
                 compute(i == 5 && (i == 4 || check(NoReturnDtor()))))) ||
        i != 4) {
      clang_analyzer_eval(true);  // expected-warning{{TRUE}}
    }

    if (compute(i == 5 &&
                (i == 4 || i == 4 ||
                 compute(i == 5 && (i == 4 || check(NoReturnDtor()))))) ||
        i != 4) {
      clang_analyzer_eval(true);  // no warning, unreachable code
    }
  }

  // PR16664 and PR18159
  void testConsistencyNestedSimple(bool value) {
    if (value) {
      if (!value || check(NoReturnDtor())) {
        clang_analyzer_eval(true); // no warning, unreachable code
      }
    }
  }

  // PR16664 and PR18159
  void testConsistencyNestedComplex(bool value) {
    if (value) {
      if (!value || !value || check(NoReturnDtor())) {
        clang_analyzer_eval(true);  // no warning, unreachable code
      }
    }
  }

  // PR16664 and PR18159
  void testConsistencyNestedWarning(bool value) {
    if (value) {
      if (!value || value || check(NoReturnDtor())) {
        clang_analyzer_eval(true); // expected-warning{{TRUE}}
      }
    }
  }
  // PR16664 and PR18159
  void testConsistencyNestedComplexMidBranch(bool value) {
    if (value) {
      if (!value || !value || check(NoReturnDtor()) || value) {
        clang_analyzer_eval(true);  // no warning, unreachable code
      }
    }
  }

  // PR16664 and PR18159
  void testConsistencyNestedComplexNestedBranch(bool value) {
    if (value) {
      if (!value || (!value || check(NoReturnDtor()) || value)) {
        clang_analyzer_eval(true);  // no warning, unreachable code
      }
    }
  }

  // PR16664 and PR18159
  void testConsistencyNestedVariableModification(bool value) {
    bool other = true;
    if (value) {
      if (!other || !value || (other = false) || check(NoReturnDtor()) ||
          !other) {
        clang_analyzer_eval(true);  // no warning, unreachable code
      }
    }
  }

  void testTernaryNoReturnTrueBranch(bool value) {
    if (value) {
      bool b = value && (value ? check(NoReturnDtor()) : true);
      clang_analyzer_eval(true);  // no warning, unreachable code
    }
  }
  void testTernaryNoReturnFalseBranch(bool value) {
    if (value) {
      bool b = !value && !value ? true : check(NoReturnDtor());
      clang_analyzer_eval(true);  // no warning, unreachable code
    }
  }
  void testTernaryIgnoreNoreturnBranch(bool value) {
    if (value) {
      bool b = !value && !value ? check(NoReturnDtor()) : true;
      clang_analyzer_eval(true);  // expected-warning{{TRUE}}
    }
  }
  void testTernaryTrueBranchReached(bool value) {
    value ? clang_analyzer_warnIfReached() : // expected-warning{{REACHABLE}}
            check(NoReturnDtor());
  }
  void testTernaryFalseBranchReached(bool value) {
    value ? check(NoReturnDtor()) :
            clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }

  void testLoop() {
    for (int i = 0; i < 10; ++i) {
      if (i < 3 && (i >= 2 || check(NoReturnDtor()))) {
        clang_analyzer_eval(true);  // no warning, unreachable code
      }
    }
  }

  bool testRecursiveFrames(bool isInner) {
    if (isInner ||
        (clang_analyzer_warnIfReached(), false) || // expected-warning{{REACHABLE}}
        check(NoReturnDtor()) ||
        testRecursiveFrames(true)) {
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
    }
  }
  void testRecursiveFramesStart() { testRecursiveFrames(false); }

  void testLambdas() {
    []() { check(NoReturnDtor()); } != nullptr || check(Dtor());
  }

  void testGnuExpressionStatements(int v) {
    ({ ++v; v == 10 || check(NoReturnDtor()); v == 42; }) || v == 23;
    clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}

    ({ ++v; check(NoReturnDtor()); v == 42; }) || v == 23;
    clang_analyzer_warnIfReached();  // no warning, unreachable code
  }

  void testGnuExpressionStatementsDestructionPoint(int v) {
    // In normal context, the temporary destructor runs at the end of the full
    // statement, thus the last statement is reached.
    (++v, check(NoReturnDtor()), v == 42),
        clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}

    // GNU expression statements execute temporary destructors within the
    // blocks, thus the last statement is not reached.
    ({ ++v; check(NoReturnDtor()); v == 42; }),
        clang_analyzer_warnIfReached();  // no warning, unreachable code
  }

  void testMultipleTemporaries(bool value) {
    if (value) {
      // FIXME: Find a way to verify construction order.
      // ~Dtor should run before ~NoReturnDtor() because construction order is
      // guaranteed by comma operator.
      if (!value || check((NoReturnDtor(), Dtor())) || value) {
        clang_analyzer_eval(true);  // no warning, unreachable code
      }
    }
  }

  void testBinaryOperatorShortcut(bool value) {
    if (value) {
      if (false && false && check(NoReturnDtor()) && true) {
        clang_analyzer_eval(true);
      }
    }
  }

  void testIfAtEndOfLoop() {
    int y = 0;
    while (true) {
      if (y > 0) {
        clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
      }
      ++y;
      // Test that the CFG gets hooked up correctly when temporary destructors
      // are handled after a statically known branch condition.
      if (true) (void)0; else (void)check(NoReturnDtor());
    }
  }

  void testTernaryAtEndOfLoop() {
    int y = 0;
    while (true) {
      if (y > 0) {
        clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
      }
      ++y;
      // Test that the CFG gets hooked up correctly when temporary destructors
      // are handled after a statically known branch condition.
      true ? (void)0 : (void)check(NoReturnDtor());
    }
  }

  void testNoReturnInComplexCondition() {
    check(Dtor()) &&
        (check(NoReturnDtor()) || check(NoReturnDtor())) && check(Dtor());
    clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
  }

  void testSequencingOfConditionalTempDtors(bool b) {
    b || (check(Dtor()), check(NoReturnDtor()));
    clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
  }

  void testSequencingOfConditionalTempDtors2(bool b) {
    (b || check(Dtor())), check(NoReturnDtor());
    clang_analyzer_warnIfReached();  // no warning, unreachable code
  }

  void testSequencingOfConditionalTempDtorsWithinBinaryOperators(bool b) {
    b || (check(Dtor()) + check(NoReturnDtor()));
    clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
  }

  void f(Dtor d = Dtor());
  void testDefaultParameters() {
    f();
  }

  struct DefaultParam {
    DefaultParam(int, const Dtor& d = Dtor());
    ~DefaultParam();
  };
  void testDefaultParamConstructorsInLoops() {
    while (true) {
      // FIXME: This exact pattern triggers the temporary cleanup logic
      // to fail when adding a 'clean' state.
      DefaultParam(42);
      DefaultParam(42);
    }
  }
  void testDefaultParamConstructorsInTernariesInLoops(bool value) {
    while (true) {
      // FIXME: This exact pattern triggers the temporary cleanup logic
      // to visit the bind-temporary logic with a state that already has that
      // temporary marked as executed.
      value ? DefaultParam(42) : DefaultParam(42);
    }
  }
#else // !TEMPORARY_DTORS

// Test for fallback logic that conservatively stops exploration after
// executing a temporary constructor for a class with a no-return destructor
// when temporary destructors are not enabled in the CFG.

  struct CtorWithNoReturnDtor {
    CtorWithNoReturnDtor() = default;

    CtorWithNoReturnDtor(int x) {
      clang_analyzer_checkInlined(false); // no-warning
    }

    ~CtorWithNoReturnDtor() __attribute__((noreturn));
  };

  void testDefaultContructorWithNoReturnDtor() {
    CtorWithNoReturnDtor();
    clang_analyzer_warnIfReached();  // no-warning
  }

  void testLifeExtensionWithNoReturnDtor() {
    const CtorWithNoReturnDtor &c = CtorWithNoReturnDtor();

    // This represents an (expected) loss of coverage, since the destructor
    // of the lifetime-exended temporary is executed at the end of
    // scope.
    clang_analyzer_warnIfReached();  // no-warning
  }

#if __cplusplus >= 201103L
  CtorWithNoReturnDtor returnNoReturnDtor() {
    return {1}; // no-crash
  }
#endif

#endif // TEMPORARY_DTORS
}

void testStaticMaterializeTemporaryExpr() {
  static const Trivial &ref = getTrivial();
  clang_analyzer_eval(ref.value == 42); // expected-warning{{TRUE}}

  static const Trivial &directRef = Trivial(42);
  clang_analyzer_eval(directRef.value == 42); // expected-warning{{TRUE}}

#if __has_feature(cxx_thread_local)
  thread_local static const Trivial &threadRef = getTrivial();
  clang_analyzer_eval(threadRef.value == 42); // expected-warning{{TRUE}}

  thread_local static const Trivial &threadDirectRef = Trivial(42);
  clang_analyzer_eval(threadDirectRef.value == 42); // expected-warning{{TRUE}}
#endif
}

namespace PR16629 {
  struct A {
    explicit A(int* p_) : p(p_) {}
    int* p;
  };

  extern void escape(const A*[]);
  extern void check(int);

  void callEscape(const A& a) {
    const A* args[] = { &a };
    escape(args);
  }

  void testNoWarning() {
    int x;
    callEscape(A(&x));
    check(x); // Analyzer used to give a "x is uninitialized warning" here
  }

  void set(const A*a[]) {
    *a[0]->p = 47;
  }

  void callSet(const A& a) {
    const A* args[] = { &a };
    set(args);
  }

  void testConsistency() {
    int x;
    callSet(A(&x));
    clang_analyzer_eval(x == 47); // expected-warning{{TRUE}}
  }
}

namespace PR32088 {
  void testReturnFromStmtExprInitializer() {
    // We shouldn't try to destroy the object pointed to by `obj' upon return.
    const NonTrivial &obj = ({
      return; // no-crash
      NonTrivial(42);
    });
  }
}

namespace CopyToTemporaryCorrectly {
class Super {
public:
  void m() {
    mImpl();
  }
  virtual void mImpl() = 0;
};
class Sub : public Super {
public:
  Sub(const int &p) : j(p) {}
  virtual void mImpl() override {
    // Used to be undefined pointer dereference because we didn't copy
    // the subclass data (j) to the temporary object properly.
    (void)(j + 1); // no-warning
    if (j != 22) {
      clang_analyzer_warnIfReached(); // no-warning
    }
  }
  const int &j;
};
void run() {
  int i = 22;
  Sub(i).m();
}
}

namespace test_return_temporary {
class C {
  int x, y;

public:
  C(int x, int y) : x(x), y(y) {}
  int getX() const { return x; }
  int getY() const { return y; }
  ~C() {}
};

class D: public C {
public:
  D() : C(1, 2) {}
  D(const D &d): C(d.getX(), d.getY()) {}
};

C returnTemporaryWithVariable() { C c(1, 2); return c; }
C returnTemporaryWithAnotherFunctionWithVariable() {
  return returnTemporaryWithVariable();
}
C returnTemporaryWithCopyConstructionWithVariable() {
  return C(returnTemporaryWithVariable());
}

C returnTemporaryWithConstruction() { return C(1, 2); }
C returnTemporaryWithAnotherFunctionWithConstruction() {
  return returnTemporaryWithConstruction();
}
C returnTemporaryWithCopyConstructionWithConstruction() {
  return C(returnTemporaryWithConstruction());
}

D returnTemporaryWithVariableAndNonTrivialCopy() { D d; return d; }
D returnTemporaryWithAnotherFunctionWithVariableAndNonTrivialCopy() {
  return returnTemporaryWithVariableAndNonTrivialCopy();
}
D returnTemporaryWithCopyConstructionWithVariableAndNonTrivialCopy() {
  return D(returnTemporaryWithVariableAndNonTrivialCopy());
}

#if __cplusplus >= 201103L
C returnTemporaryWithBraces() { return {1, 2}; }
C returnTemporaryWithAnotherFunctionWithBraces() {
  return returnTemporaryWithBraces();
}
C returnTemporaryWithCopyConstructionWithBraces() {
  return C(returnTemporaryWithBraces());
}
#endif // C++11

void test() {
  C c1 = returnTemporaryWithVariable();
  clang_analyzer_eval(c1.getX() == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(c1.getY() == 2); // expected-warning{{TRUE}}

  C c2 = returnTemporaryWithAnotherFunctionWithVariable();
  clang_analyzer_eval(c2.getX() == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(c2.getY() == 2); // expected-warning{{TRUE}}

  C c3 = returnTemporaryWithCopyConstructionWithVariable();
  clang_analyzer_eval(c3.getX() == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(c3.getY() == 2); // expected-warning{{TRUE}}

  C c4 = returnTemporaryWithConstruction();
  clang_analyzer_eval(c4.getX() == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(c4.getY() == 2); // expected-warning{{TRUE}}

  C c5 = returnTemporaryWithAnotherFunctionWithConstruction();
  clang_analyzer_eval(c5.getX() == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(c5.getY() == 2); // expected-warning{{TRUE}}

  C c6 = returnTemporaryWithCopyConstructionWithConstruction();
  clang_analyzer_eval(c5.getX() == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(c5.getY() == 2); // expected-warning{{TRUE}}

#if __cplusplus >= 201103L

  C c7 = returnTemporaryWithBraces();
  clang_analyzer_eval(c7.getX() == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(c7.getY() == 2); // expected-warning{{TRUE}}

  C c8 = returnTemporaryWithAnotherFunctionWithBraces();
  clang_analyzer_eval(c8.getX() == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(c8.getY() == 2); // expected-warning{{TRUE}}

  C c9 = returnTemporaryWithCopyConstructionWithBraces();
  clang_analyzer_eval(c9.getX() == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(c9.getY() == 2); // expected-warning{{TRUE}}

#endif // C++11

  D d1 = returnTemporaryWithVariableAndNonTrivialCopy();
  clang_analyzer_eval(d1.getX() == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(d1.getY() == 2); // expected-warning{{TRUE}}

  D d2 = returnTemporaryWithAnotherFunctionWithVariableAndNonTrivialCopy();
  clang_analyzer_eval(d2.getX() == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(d2.getY() == 2); // expected-warning{{TRUE}}

  D d3 = returnTemporaryWithCopyConstructionWithVariableAndNonTrivialCopy();
  clang_analyzer_eval(d3.getX() == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(d3.getY() == 2); // expected-warning{{TRUE}}
}
} // namespace test_return_temporary


namespace test_temporary_object_expr_without_dtor {
class C {
  int x;
public:
  C(int x) : x(x) {}
  int getX() const { return x; }
};

void test() {
  clang_analyzer_eval(C(3).getX() == 3); // expected-warning{{TRUE}}
};
}

namespace test_temporary_object_expr_with_dtor {
class C {
  int x;

public:
  C(int x) : x(x) {}
  ~C() {}
  int getX() const { return x; }
};

void test(int coin) {
  clang_analyzer_eval(C(3).getX() == 3);
#ifdef TEMPORARY_DTORS
  // expected-warning@-2{{TRUE}}
#else
  // expected-warning@-4{{UNKNOWN}}
#endif

  const C &c1 = coin ? C(1) : C(2);
  if (coin) {
    clang_analyzer_eval(c1.getX() == 1);
#ifdef TEMPORARY_DTORS
  // expected-warning@-2{{TRUE}}
#else
  // expected-warning@-4{{UNKNOWN}}
#endif
  } else {
    clang_analyzer_eval(c1.getX() == 2);
#ifdef TEMPORARY_DTORS
  // expected-warning@-2{{TRUE}}
#else
  // expected-warning@-4{{UNKNOWN}}
#endif
  }

  C c2 = coin ? C(1) : C(2);
  if (coin) {
    clang_analyzer_eval(c2.getX() == 1); // expected-warning{{TRUE}}
  } else {
    clang_analyzer_eval(c2.getX() == 2); // expected-warning{{TRUE}}
  }
}

} // namespace test_temporary_object_expr

namespace test_match_constructors_and_destructors {
class C {
public:
  int &x, &y;
  C(int &_x, int &_y) : x(_x), y(_y) { ++x; }
  C(const C &c): x(c.x), y(c.y) { ++x; }
  ~C() { ++y; }
};

void test_simple_temporary() {
  int x = 0, y = 0;
  {
    const C &c = C(x, y);
  }
  // One constructor and one destructor.
  clang_analyzer_eval(x == 1);
  clang_analyzer_eval(y == 1);
#ifdef TEMPORARY_DTORS
  // expected-warning@-3{{TRUE}}
  // expected-warning@-3{{TRUE}}
#else
  // expected-warning@-6{{UNKNOWN}}
  // expected-warning@-6{{UNKNOWN}}
#endif
}

void test_simple_temporary_with_copy() {
  int x = 0, y = 0;
  {
    C c = C(x, y);
  }
  // Only one constructor directly into the variable, and one destructor.
  clang_analyzer_eval(x == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(y == 1); // expected-warning{{TRUE}}
}

void test_ternary_temporary(int coin) {
  int x = 0, y = 0, z = 0, w = 0;
  {
    const C &c = coin ? C(x, y) : C(z, w);
  }
  // Only one constructor on every branch, and one automatic destructor.
  if (coin) {
    clang_analyzer_eval(x == 1);
    clang_analyzer_eval(y == 1);
#ifdef TEMPORARY_DTORS
    // expected-warning@-3{{TRUE}}
    // expected-warning@-3{{TRUE}}
#else
    // expected-warning@-6{{UNKNOWN}}
    // expected-warning@-6{{UNKNOWN}}
#endif
    clang_analyzer_eval(z == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(w == 0); // expected-warning{{TRUE}}

  } else {
    clang_analyzer_eval(x == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(y == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(z == 1);
    clang_analyzer_eval(w == 1);
#ifdef TEMPORARY_DTORS
    // expected-warning@-3{{TRUE}}
    // expected-warning@-3{{TRUE}}
#else
    // expected-warning@-6{{UNKNOWN}}
    // expected-warning@-6{{UNKNOWN}}
#endif
  }
}

void test_ternary_temporary_with_copy(int coin) {
  int x = 0, y = 0, z = 0, w = 0;
  {
    C c = coin ? C(x, y) : C(z, w);
  }
  // On each branch the variable is constructed directly.
  if (coin) {
    clang_analyzer_eval(x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(y == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(z == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(w == 0); // expected-warning{{TRUE}}

  } else {
    clang_analyzer_eval(x == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(y == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(z == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(w == 1); // expected-warning{{TRUE}}
  }
}
} // namespace test_match_constructors_and_destructors

namespace destructors_for_return_values {

class C {
public:
  ~C() {
    1 / 0; // expected-warning{{Division by zero}}
  }
};

C make();

void testFloatingCall() {
  make();
  // Should have divided by zero in the destructor.
  clang_analyzer_warnIfReached();
#ifndef TEMPORARY_DTORS
    // expected-warning@-2{{REACHABLE}}
#endif
}

void testLifetimeExtendedCall() {
  {
    const C &c = make();
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
  // Should have divided by zero in the destructor.
  clang_analyzer_warnIfReached(); // no-warning
}

void testCopiedCall() {
  {
    C c = make();
    // Should have elided the constructor/destructor for the temporary
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
  // Should have divided by zero in the destructor.
  clang_analyzer_warnIfReached(); // no-warning
}
} // namespace destructors_for_return_values

namespace dont_forget_destructor_around_logical_op {
int glob;

class C {
public:
  ~C() {
    glob = 1;
    clang_analyzer_checkInlined(true);
#ifdef TEMPORARY_DTORS
    // expected-warning@-2{{TRUE}}
#endif
  }
};

C get();

bool is(C);


void test(int coin) {
  // Here temporaries are being cleaned up after && is evaluated. There are two
  // temporaries: the return value of get() and the elidable copy constructor
  // of that return value into is(). According to the CFG, we need to cleanup
  // both of them depending on whether the temporary corresponding to the
  // return value of get() was initialized. However, we didn't track
  // temporaries returned from functions, so we took the wrong branch.
  coin && is(get()); // no-crash
  if (coin) {
    clang_analyzer_eval(glob);
#ifdef TEMPORARY_DTORS
    // expected-warning@-2{{TRUE}}
#else
    // expected-warning@-4{{UNKNOWN}}
#endif
  } else {
    // The destructor is not called on this branch.
    clang_analyzer_eval(glob); // expected-warning{{UNKNOWN}}
  }
}
} // namespace dont_forget_destructor_around_logical_op

#if __cplusplus >= 201103L
namespace temporary_list_crash {
class C {
public:
  C() {}
  ~C() {}
};

void test() {
  std::initializer_list<C>{C(), C()}; // no-crash
}
} // namespace temporary_list_crash
#endif // C++11

namespace implicit_constructor_conversion {
struct S {
  int x;
  S(int x) : x(x) {}
  ~S() {}
};

class C {
  int x;

public:
  C(const S &s) : x(s.x) {}
  ~C() {}
  int getX() const { return x; }
};

void test() {
  const C &c1 = S(10);
  clang_analyzer_eval(c1.getX() == 10);
#ifdef TEMPORARY_DTORS
  // expected-warning@-2{{TRUE}}
#else
  // expected-warning@-4{{UNKNOWN}}
#endif

  S s = 20;
  clang_analyzer_eval(s.x == 20); // expected-warning{{TRUE}}

  C c2 = s;
  clang_analyzer_eval(c2.getX() == 20); // expected-warning{{TRUE}}
}
} // end namespace implicit_constructor_conversion

namespace pass_references_through {
class C {
public:
  ~C() {}
};

const C &foo1();
C &&foo2();

// In these examples the foo() expression has record type, not reference type.
// Don't try to figure out how to perform construction of the record here.
const C &bar1() { return foo1(); } // no-crash
C &&bar2() { return foo2(); } // no-crash
} // end namespace pass_references_through
