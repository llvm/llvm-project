// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify -w -std=c++03 %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify -w -std=c++11 %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -DTEMPORARY_DTORS -verify -w -analyzer-config cfg-temporary-dtors=true %s -std=c++11

extern bool clang_analyzer_eval(bool);
extern bool clang_analyzer_warnIfReached();

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

    ~CtorWithNoReturnDtor() __attribute__((noreturn));
  };

  void testDefaultContructorWithNoReturnDtor() {
    CtorWithNoReturnDtor();
    clang_analyzer_warnIfReached();  // no-warning
  }

  void testLifeExtensionWithNoReturnDtor() {
    const CtorWithNoReturnDtor &c = CtorWithNoReturnDtor();

    // This represents an (expected) loss of coverage, since the destructor
    // of the lifetime-exended temporary is executed at at the end of
    // scope.
    clang_analyzer_warnIfReached();  // no-warning
  }

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
