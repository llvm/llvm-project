// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.cplusplus.LifetimeAnnotations,debug.DebugLifetimeAnnotations \
// RUN:   -analyzer-config cfg-lifetime=true -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.cplusplus.LifetimeAnnotations,debug.DebugLifetimeAnnotations \
// RUN:   -analyzer-config c++-container-inlining=false -analyzer-config cfg-lifetime=true -verify %s

struct A {};

void clang_analyzer_lifetime_bound(int*);
void clang_analyzer_lifetime_bound(int&);
void clang_analyzer_lifetime_bound(A*);
void clang_analyzer_lifetime_bound(A&);

// These are the cases when the result of function calls are MemRegions.

// Ref type parameter annotated case
struct X {
  int& choose(int& a [[clang::lifetimebound]]) { return a; }
};

void caller() {
  int v = 0;
  X obj;
  int& r = obj.choose(v);
  clang_analyzer_lifetime_bound(r); // expected-warning {{Origin v bound to v}}
}

// Obj ref type function return annotated case
struct Y {
  A a;
  A& getA() [[clang::lifetimebound]] { return a; }
};

void caller_two() {
  // Return statement is annotated case.
  Y y;
  A& f = y.getA();
  clang_analyzer_lifetime_bound(f); // expected-warning {{Origin y.a bound to y}}
}

// Obj ptr type function return annotated case
struct Z {
  A a;
  A* getA() [[clang::lifetimebound]] { return &a; }
};

void caller_three() {
  Z z;
  A* func = z.getA();
  clang_analyzer_lifetime_bound(func); // expected-warning {{Origin z.a bound to z}}
}

// Free function with annotated param and ref return
int& foo(int& num [[clang::lifetimebound]]) { return num; }

void caller_four() {
  int num = 5;
  int& s = foo(num);
  clang_analyzer_lifetime_bound(s); // expected-warning {{Origin num bound to num}}
}

// Free function with annotated param and ptr return
int* boo(int* num [[clang::lifetimebound]]) { return num; }

void caller_five() {
  int n = 55;
  int* n_ptr = &n;
  int* s = boo(n_ptr);

  clang_analyzer_lifetime_bound(s); // expected-warning {{Origin n bound to n}}
}

// Free function with both annotated and non-annotated parameters.
int& fn(int& f, int& s [[clang::lifetimebound]]) { return s; }

void caller_six() {
  int even = 50;
  int odd = 55;
  int& s = fn(even, odd);

  clang_analyzer_lifetime_bound(s); // expected-warning {{Origin odd bound to odd}}
}



// These are the cases when the result of function calls are SymbolRefs.

// Function returns ptr and has an annotated parameter
int* foo(int* n [[clang::lifetimebound]]);

void caller_seven() {
  int y = 15;
  int* y_ptr = &y;
  auto* bind = foo(y_ptr);

  clang_analyzer_lifetime_bound(bind); // expected-warning-re {{Origin conj_${{[0-9]+}}{int *, LC{{[0-9]+}}, S{{[0-9]+}}, #{{[0-9]+}}} bound to y}}
}

// Function returns a reference and has an annotated parameter
int& func(int& some_number [[clang::lifetimebound]]);

void caller_eight() {
  int f = 15;
  auto& bind = func(f);

  clang_analyzer_lifetime_bound(bind); // expected-warning-re {{Origin conj_${{[0-9]+}}{int &, LC{{[0-9]+}}, S{{[0-9]+}}, #{{[0-9]+}}} bound to f}}
}

// Function returns a reference and has two annotated parameters.
int& f(int& a [[clang::lifetimebound]], int& b [[clang::lifetimebound]]);

void caller_nine() {
  int first_num = 1;
  int second_num = 2;
  int& numbers = f(first_num, second_num);

  clang_analyzer_lifetime_bound(numbers);
  // expected-warning-re@-1 {{Origin conj_${{[0-9]+}}{int &, LC{{[0-9]+}}, S{{[0-9]+}}, #{{[0-9]+}}} bound to first_num}}
  // expected-warning-re@-2 {{Origin conj_${{[0-9]+}}{int &, LC{{[0-9]+}}, S{{[0-9]+}}, #{{[0-9]+}}} bound to second_num}}
}

struct View {
  int* p;
};
View makeView(int& x [[clang::lifetimebound]]);

void clang_analyzer_lifetime_bound(View);

void caller_view() {
  int v = 42;
  View w = makeView(v);
  // FIXME: Currently none of the maps cover LazyCompoundVal
  clang_analyzer_lifetime_bound(w); // no-warning
}



// These are the test cases for testing the correctness of the emitted warning from the LifetimeAnnotations checker.

// Return value bound to annotated param cases
int *test_func(int *p [[clang::lifetimebound]]);


int *direct_return() {
  int i = 5;
  return test_func(&i);
  // expected-warning@-1 {{Returning value bound to a local 'i' that will go out of scope}}
  // expected-warning@-2 {{address of stack memory associated with local variable 'i' returned}}
}

int *variable_return() {
  int y = 5;
  int *p = test_func(&y);
  return p; // expected-warning {{Returning value bound to a local 'y' that will go out of scope}}
}

int *borrow_from_caller(int *b [[clang::lifetimebound]]) {
  return test_func(b); // no-warning
}

void no_return() {
  int i = 5;
  int *p = test_func(&i);
  (void)p; // no-warning
}

// Use-after-scope dangling pointer dereference
void caller_ten() {
  int* p = nullptr;
  {
    int x = 1;
    p = test_func(&x);
  }
  *p = 2; // expected-warning {{Use of 'x' after its lifetime ended}}
}

void out_of_scope_ptr() {
  int *ptr = nullptr;
  {
    int n = 5;
    ptr = &n;
  }
  *ptr = 3; // expected-warning {{Use of 'n' after its lifetime ended}}
}

void f() {
  int* p;
  {
    int x = 1;
    p = &x;
  }
  int y = 2;
  p = &y;
  *p = 3; // no-warning
}
