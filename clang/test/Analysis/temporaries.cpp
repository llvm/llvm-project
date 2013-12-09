// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -verify -w -std=c++03 %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -verify -w -std=c++11 %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -DTEMPORARY_DTORS -verify -w -analyzer-config cfg-temporary-dtors=true %s 

extern bool clang_analyzer_eval(bool);

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
  void testPR16664andPR18159Crash() {
    struct Dtor {
      ~Dtor();
    };
    extern bool coin();
    extern bool check(const Dtor &);

#ifndef TEMPORARY_DTORS
    // FIXME: Don't crash here when tmp dtros are enabled.
    // PR16664 and PR18159 
    if (coin() && (coin() || coin() || check(Dtor()))) {
      Dtor();
    }
#endif
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


/*
  // PR16664 and PR18159 
  FIXME: Don't crash here.
  void testConsistencyNested(int i) {
    extern bool compute(bool);
  
    if (i == 5 && (i == 4 || i == 5 || check(NoReturnDtor())))
      clang_analyzer_eval(true); // expected TRUE
  
    if (i == 5 && (i == 4 || i == 5 || check(NoReturnDtor())))
      clang_analyzer_eval(true);  // expected TRUE

    if (i != 5)
      return;

    if (compute(i == 5 &&
                (i == 4 || compute(true) ||
                 compute(i == 5 && (i == 4 || check(NoReturnDtor()))))) ||
        i != 4) {
      clang_analyzer_eval(true); // expected TRUE
    }

    if (compute(i == 5 &&
                (i == 4 || i == 4 ||
                 compute(i == 5 && (i == 4 || check(NoReturnDtor()))))) ||
        i != 4) {
      clang_analyzer_eval(true); // no warning, unreachable code
    }
  }*/
  
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
