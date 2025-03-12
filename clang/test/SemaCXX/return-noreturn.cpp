// RUN: %clang_cc1 %s -fsyntax-only -fcxx-exceptions -verify -Wreturn-type -Wno-error=return-type -Wmissing-noreturn -Wno-unreachable-code -Wno-covered-switch-default
// RUN: %clang_cc1 %s -fsyntax-only -fcxx-exceptions -std=c++11 -verify -Wreturn-type -Wno-error=return-type -Wmissing-noreturn -Wno-unreachable-code -Wno-covered-switch-default

// A destructor may be marked noreturn and should still influence the CFG.
void pr6884_abort() __attribute__((noreturn));

struct pr6884_abort_struct {
  pr6884_abort_struct() {}
  ~pr6884_abort_struct() __attribute__((noreturn)) { pr6884_abort(); }
};

struct other { ~other() {} };

// Ensure that destructors from objects are properly modeled in the CFG despite
// the presence of switches, case statements, labels, and blocks. These tests
// try to cover bugs reported in both PR6884 and PR10063.
namespace abort_struct_complex_cfgs {
  int basic(int x) {
    switch (x) { default: pr6884_abort(); }
  }
  int f1(int x) {
    switch (x) default: pr6884_abort_struct();
  }
  int f2(int x) {
    switch (x) { default: pr6884_abort_struct(); }
  }
  int f2_positive(int x) {
    switch (x) { default: ; }
  } // expected-warning {{non-void function does not return a value}}
  int f3(int x) {
    switch (x) { default: { pr6884_abort_struct(); } }
  }
  int f4(int x) {
    switch (x) default: L1: L2: case 4: pr6884_abort_struct();
  }
  int f5(int x) {
    switch (x) default: L1: { L2: case 4: pr6884_abort_struct(); }
  }
  int f6(int x) {
    switch (x) default: L1: L2: case 4: { pr6884_abort_struct(); }
  }

  // FIXME: detect noreturn destructors triggered by calls to delete.
  int f7(int x) {
    switch (x) default: L1: L2: case 4: {
      pr6884_abort_struct *p = new pr6884_abort_struct();
      delete p;
    }
  } // expected-warning {{non-void function does not return a value}}

  // Test that these constructs work even when extraneous blocks are created
  // before and after the switch due to implicit destructors.
  int g1(int x) {
    other o;
    switch (x) default: pr6884_abort_struct();
  }
  int g2(int x) {
    other o;
    switch (x) { default: pr6884_abort_struct(); }
  }
  int g2_positive(int x) {
    other o;
    switch (x) { default: ; }
  } // expected-warning {{non-void function does not return a value}}
  int g3(int x) {
    other o;
    switch (x) { default: { pr6884_abort_struct(); } }
  }
  int g4(int x) {
    other o;
    switch (x) default: L1: L2: case 4: pr6884_abort_struct();
  }
  int g5(int x) {
    other o;
    switch (x) default: L1: { L2: case 4: pr6884_abort_struct(); }
  }
  int g6(int x) {
    other o;
    switch (x) default: L1: L2: case 4: { pr6884_abort_struct(); }
  }

  // Test that these constructs work even with variables carrying the no-return
  // destructor instead of temporaries.
  int h1(int x) {
    other o;
    switch (x) default: pr6884_abort_struct a;
  }
  int h2(int x) {
    other o;
    switch (x) { default: pr6884_abort_struct a; }
  }
  int h3(int x) {
    other o;
    switch (x) { default: { pr6884_abort_struct a; } }
  }
  int h4(int x) {
    other o;
    switch (x) default: L1: L2: case 4: pr6884_abort_struct a;
  }
  int h5(int x) {
    other o;
    switch (x) default: L1: { L2: case 4: pr6884_abort_struct a; }
  }
  int h6(int x) {
    other o;
    switch (x) default: L1: L2: case 4: { pr6884_abort_struct a; }
  }
}

// PR9380
struct PR9380 {
  ~PR9380();
};
struct PR9380_B : public PR9380 {
  PR9380_B( const PR9380& str );
};
void test_PR9380(const PR9380& aKey) {
  const PR9380& flatKey = PR9380_B(aKey);
}

// Array of objects with destructors.  This is purely a coverage test case.
void test_array() {
  PR9380 a[2];
}

// Test classes wrapped in typedefs.  This is purely a coverage test case
// for CFGImplictDtor::getDestructorDecl().
void test_typedefs() {
  typedef PR9380 PR9380_Ty;
  PR9380_Ty test;
  PR9380_Ty test2[20];
}

// PR9412 - Handle CFG traversal with null successors.
enum PR9412_MatchType { PR9412_Exact };

template <PR9412_MatchType type> int PR9412_t() {
  switch (type) {
    case PR9412_Exact:
    default:
        break;
  }
} // expected-warning {{non-void function does not return a value}}

void PR9412_f() {
    PR9412_t<PR9412_Exact>(); // expected-note {{in instantiation of function template specialization 'PR9412_t<PR9412_Exact>' requested here}}
}

struct NoReturn {
  ~NoReturn() __attribute__((noreturn));
  operator bool() const;
};
struct Return {
  ~Return();
  operator bool() const;
};

int testTernaryUnconditionalNoreturn() {
  true ? NoReturn() : NoReturn();
}

int testTernaryStaticallyConditionalNoretrunOnTrue() {
  true ? NoReturn() : Return();
}

int testTernaryStaticallyConditionalRetrunOnTrue() {
  true ? Return() : NoReturn();
} // expected-warning {{non-void function does not return a value}}

int testTernaryStaticallyConditionalNoretrunOnFalse() {
  false ? Return() : NoReturn();
}

int testTernaryStaticallyConditionalRetrunOnFalse() {
  false ? NoReturn() : Return();
} // expected-warning {{non-void function does not return a value}}

int testTernaryConditionalNoreturnTrueBranch(bool value) {
  value ? (NoReturn() || NoReturn()) : Return();
} // expected-warning {{non-void function does not return a value in all control paths}}

int testTernaryConditionalNoreturnFalseBranch(bool value) {
  value ? Return() : (NoReturn() || NoReturn());
} // expected-warning {{non-void function does not return a value in all control paths}}

int testConditionallyExecutedComplexTernaryTrueBranch(bool value) {
  value || (true ? NoReturn() : true);
} // expected-warning {{non-void function does not return a value in all control paths}}

int testConditionallyExecutedComplexTernaryFalseBranch(bool value) {
  value || (false ? true : NoReturn());
} // expected-warning {{non-void function does not return a value in all control paths}}

int testStaticallyExecutedLogicalOrBranch() {
  false || NoReturn();
}

int testStaticallyExecutedLogicalAndBranch() {
  true && NoReturn();
}

int testStaticallySkippedLogicalOrBranch() {
  true || NoReturn();
} // expected-warning {{non-void function does not return a value}}

int testStaticallySkppedLogicalAndBranch() {
  false && NoReturn();
} // expected-warning {{non-void function does not return a value}}

int testConditionallyExecutedComplexLogicalBranch(bool value) {
  value || (true && NoReturn());
} // expected-warning {{non-void function does not return a value in all control paths}}

int testConditionallyExecutedComplexLogicalBranch2(bool value) {
  (true && value) || (true && NoReturn());
} // expected-warning {{non-void function does not return a value in all control paths}}

int testConditionallyExecutedComplexLogicalBranch3(bool value) {
  (false && (Return() || true)) || (true && NoReturn());
}

int testConditionallyExecutedComplexLogicalBranch4(bool value) {
  false || ((Return() || true) && (true && NoReturn()));
}

#if __cplusplus >= 201103L
namespace LambdaVsTemporaryDtor {
  struct Y { ~Y(); };
  struct X { template<typename T> X(T, Y = Y()) {} };

  struct Fatal { ~Fatal() __attribute__((noreturn)); };
  struct FatalCopy { FatalCopy(); FatalCopy(const FatalCopy&, Fatal F = Fatal()); };

  void foo();

  int bar() {
    X work([](){ Fatal(); });
    foo();
  } // expected-warning {{non-void function does not return a value}}

  int baz() {
    FatalCopy fc;
    X work([fc](){});
    foo();
  } // ok, initialization of lambda does not return
}
#endif

// Ensure that function-try-blocks also check for return values properly.
int functionTryBlock1(int s) try {
  return 0;
} catch (...) {
} // expected-warning {{non-void function does not return a value in all control paths}}

int functionTryBlock2(int s) try {
} catch (...) {
  return 0;
} // expected-warning {{non-void function does not return a value in all control paths}}

int functionTryBlock3(int s) try {
  return 0;
} catch (...) {
  return 0;
} // ok, both paths return.
