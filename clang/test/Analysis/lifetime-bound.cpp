// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.cplusplus.LifetimeAnnotations \
// RUN:   -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.cplusplus.LifetimeAnnotations \
// RUN:   -analyzer-config c++-container-inlining=false -verify %s

void clang_analyzer_dump(...);

// These are the cases when the result of function calls are MemRegions.

struct A {};

// Ref type parameter annotated case
struct X {
  int& choose(int& a [[clang::lifetimebound]]) { return a; }
};

void clang_analyzer_lifetime_bound(int&);

void caller() {
  int v = 0;
  X obj;
  int& r = obj.choose(v);
  clang_analyzer_lifetime_bound(r); // expected-warning {{Origin v bound to v}}
  clang_analyzer_dump(r);
}

// Obj ref type function return annotated case
struct Y {
  A a;
  A& getA() [[clang::lifetimebound]] { return a; }
};

void clang_analyzer_lifetime_bound(A& a);

void caller_two() {
  // Return statement is annotated case.
  Y y;
  A& f = y.getA();
  clang_analyzer_lifetime_bound(f); // expected-warning {{Origin y.a bound to y}}
  clang_analyzer_dump(f);
}

// Obj ptr type function return annotated case
struct Z {
  A a;
  A* getA() [[clang::lifetimebound]] { return &a; }
};

void clang_analyzer_lifetime_bound(A* a);

void caller_three() {
  Z z;
  A* func = z.getA();
  clang_analyzer_lifetime_bound(func); // expected-warning {{Origin z.a bound to z}}
  clang_analyzer_dump(func);
}

// Free function with annotated param and ref return
int& foo(int& num [[clang::lifetimebound]]) { return num; }

void clang_analyzer_lifetime_bound(int&);

void caller_four() {
  int num = 5;
  int& s = foo(num);
  clang_analyzer_lifetime_bound(s); // expected-warning {{Origin num bound to num}}
  clang_analyzer_dump(s);
}

// Free function with annotated param and ptr return
int* boo(int* num [[clang::lifetimebound]]) { return num; }

void clang_analyzer_lifetime_bound(int*);

void caller_five() {
  int n = 55;
  int* n_ptr = &n;
  int* s = boo(n_ptr);

  clang_analyzer_lifetime_bound(s); // expected-warning {{Origin n bound to n}}
  clang_analyzer_dump(s);
}

// These are the cases when the result of function calls are SymbolRefs.

// Function returns ptr and has an annotated parameter
int* foo(int* n [[clang::lifetimebound]]);

void clang_analyzer_lifetime_bound(int*);

void caller_six() {
  int y = 15;
  int* y_ptr = &y;
  auto bind = foo(y_ptr);

  clang_analyzer_lifetime_bound(bind);
                                       // expected-warning@-1 {{Origin bound to n}}
                                      // expected-warning@-1 {{Origin contains loan n}}
  clang_analyzer_dump(bind);

// FIXME: The full warning does look like this:
// Origin SymRegion{conj_$5{int *, LC1, S847, #1}} bound to n
// Origin conj_$5{int *, LC1, S847, #1} contains loan n
// Since the conj sym number and the ID can change across runs I have decided to just include
// string parts of the error message since that is the only consistent part of the emitted report.
// This does not apply to the test cases above this test case.
}


// Function returns a reference and has an annotated parameter

