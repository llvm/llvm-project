// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.cplusplus.UseAfterLifetimeEnd,debug.DebugUseAfterLifetimeEnd \
// RUN:   -analyzer-config cfg-lifetime=true -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.cplusplus.UseAfterLifetimeEnd,debug.DebugUseAfterLifetimeEnd \
// RUN:   -analyzer-config c++-container-inlining=false -analyzer-config cfg-lifetime=true -verify %s

struct A {};

void clang_analyzer_dumpLifetimeOriginsOf(int*);
void clang_analyzer_dumpLifetimeOriginsOf(int&);
void clang_analyzer_dumpLifetimeOriginsOf(A*);
void clang_analyzer_dumpLifetimeOriginsOf(A&);

// These are the cases when the result of function calls are MemRegions.

// Ref type parameter annotated case.
struct X {
  int &choose(int &a [[clang::lifetimebound]]) { return a; }
};

void caller() {
  int v = 0;
  X obj;
  int &r = obj.choose(v);
  clang_analyzer_dumpLifetimeOriginsOf(r); // expected-warning {{Origin &v bound to v}}
}

// Obj ref type function return annotated case.
struct Y {
  A a;
  A &getA() [[clang::lifetimebound]] { return a; }
};

void caller_two() {
  // Return statement is annotated case.
  Y y;
  A &f = y.getA();
  clang_analyzer_dumpLifetimeOriginsOf(f); // expected-warning {{Origin &y.a bound to y}}
}

// Obj ptr type function return annotated case.
struct Z {
  A a;
  A *getA() [[clang::lifetimebound]] { return &a; }
};

void caller_three() {
  Z z;
  A *func = z.getA();
  clang_analyzer_dumpLifetimeOriginsOf(func); // expected-warning {{Origin &z.a bound to z}}
}

// Free function with annotated param and ref return.
int &foo(int &num [[clang::lifetimebound]]) { return num; }

void caller_four() {
  int num = 5;
  int &s = foo(num);
  clang_analyzer_dumpLifetimeOriginsOf(s); // expected-warning {{Origin &num bound to num}}
}

// Free function with annotated param and ptr return.
int *boo(int *num [[clang::lifetimebound]]) { return num; }

void caller_five() {
  int n = 55;
  int *n_ptr = &n;
  int *s = boo(n_ptr);

  clang_analyzer_dumpLifetimeOriginsOf(s); // expected-warning {{Origin &n bound to n}}
}

// Free function with both annotated and non-annotated parameters.
int &fn(int &f, int &s [[clang::lifetimebound]]) { return s; }

void caller_six() {
  int even = 50;
  int odd = 55;
  int &s = fn(even, odd);

  clang_analyzer_dumpLifetimeOriginsOf(s); // expected-warning {{Origin &odd bound to odd}}
}



// These are the cases when the result of function calls are SymbolRefs.

// Function returns ptr and has an annotated parameter.
int *foo(int *n [[clang::lifetimebound]]);

void caller_seven() {
  int y = 15;
  int *y_ptr = &y;
  auto *bind = foo(y_ptr);

  clang_analyzer_dumpLifetimeOriginsOf(bind); // expected-warning-re {{Origin &SymRegion{{.*}} bound to y}}
}

// Function returns a reference and has an annotated parameter.
int &func(int &some_number [[clang::lifetimebound]]);

void caller_eight() {
  int f = 15;
  auto &bind = func(f);

  clang_analyzer_dumpLifetimeOriginsOf(bind); // expected-warning-re {{Origin &SymRegion{{.*}} bound to f}}
}

// Function returns a reference and has two annotated parameters.
int &f(int &a [[clang::lifetimebound]], int &b [[clang::lifetimebound]]);

void caller_nine() {
  int first_num = 1;
  int second_num = 2;
  int &numbers = f(first_num, second_num);

  clang_analyzer_dumpLifetimeOriginsOf(numbers); // expected-warning-re {{Origin &SymRegion{{.*}} bound to first_num, second_num}}
}

struct View {
  int *p;
};
View makeView(int &x [[clang::lifetimebound]]);

void clang_analyzer_dumpLifetimeOriginsOf(View);

void caller_view() {
  int v = 42;
  View w = makeView(v);
  // FIXME: Currently none of the maps cover LazyCompoundVal.
  clang_analyzer_dumpLifetimeOriginsOf(w); // no-warning
}



// These are the test cases for testing the correctness of the emitted warning from the UseAfterLifetimeEnd checker.

// Return value bound to annotated param cases.
int *test_func(int *p [[clang::lifetimebound]]);


int *direct_return() {
  int i = 5;
  return test_func(&i);
  // expected-warning@-1 {{Returning value bound to 'i' that will go out of scope}}
  // expected-warning@-2 {{address of stack memory associated with local variable 'i' returned}}
}

int *variable_return() {
  int y = 5;
  int *p = test_func(&y);
  return p; // expected-warning {{Returning value bound to 'y' that will go out of scope}}
}

int *borrow_from_caller(int *b [[clang::lifetimebound]]) {
  return test_func(b); // no-warning
}

void no_return() {
  int i = 5;
  int *p = test_func(&i);
  (void)p; // no-warning
}

int *g() {
  int i = 5;
  int *p = test_func(&i);
  (void)p;
  return nullptr; // no-warning
}

int &multi_param_test_ref(int &a [[clang::lifetimebound]], int &b [[clang::lifetimebound]]);

// Return value bound to annotated parameters (two dangling sources).
int &dangling_sources_ref() {
  int x = 1, y = 2;
  return multi_param_test_ref(x, y);
  // expected-warning@-1 {{Returning value bound to 'x' that will go out of scope}}
  // expected-warning@-2 {{Returning value bound to 'y' that will go out of scope}}
  // expected-warning@-3 {{reference to stack memory associated with local variable 'x' returned}}
  // expected-warning@-4 {{reference to stack memory associated with local variable 'y' returned}}
}

// Return value bound to annotated parameters (no dangling sources).
int &no_dangling_sources_ref(int &a [[clang::lifetimebound]], int &b [[clang::lifetimebound]]) {
  return multi_param_test_ref(a, b); // no-warning
}

// Return value bound to annotated parameters (one dangling source).
int &one_dangling_source_ref(int &a [[clang::lifetimebound]]) {
  int x = 1;
  return multi_param_test_ref(a, x);
  // expected-warning@-1 {{Returning value bound to 'x' that will go out of scope}}
  // expected-warning@-2 {{reference to stack memory associated with local variable 'x' returned}}
}

int *multi_param_test_ptr(int *a [[clang::lifetimebound]], int *b [[clang::lifetimebound]]);

// Return value bound to annotated parameters (two dangling sources).
int *dangling_sources_ptr() {
  int x = 1, y = 2;
  int *x_ptr = &x;
  int *y_ptr = &y;
  return multi_param_test_ptr(x_ptr, y_ptr);
  // expected-warning@-1 {{Returning value bound to 'x' that will go out of scope}}
  // expected-warning@-2 {{Returning value bound to 'y' that will go out of scope}}
}

// Return value bound to annotated parameters (no dangling sources).
int *no_dangling_sources_ptr(int *a [[clang::lifetimebound]], int *b [[clang::lifetimebound]]) {
  return multi_param_test_ptr(a, b); // no-warning
}

// Return value bound to annotated parameters (one dangling source).
int *one_dangling_source_ptr(int *a [[clang::lifetimebound]]) {
  int x = 1;
  int *x_ptr = &x;
  return multi_param_test_ptr(a, x_ptr); // expected-warning {{Returning value bound to 'x' that will go out of scope}}
}

