// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.cplusplus.LifetimeAnnotations \
// RUN:   -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.cplusplus.LifetimeAnnotations \
// RUN:   -analyzer-config c++-container-inlining=false -verify %s

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
  clang_analyzer_lifetime_bound(r); // expected-warning {{bound to v}}
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
  clang_analyzer_lifetime_bound(f); // expected-warning {{bound to y}}
}

// Obj ptr type function return annotated case
struct Z {
  A a;
  A* getA() [[clang::lifetimebound]] { return &a; }
};

void caller_three() {
  Z z;
  A* func = z.getA();
  clang_analyzer_lifetime_bound(func); // expected-warning {{bound to z}}
}

// Free function with annotated param and ref return
int& foo(int& num [[clang::lifetimebound]]) { return num; }

void caller_four() {
  int num = 5;
  int& s = foo(num);
  clang_analyzer_lifetime_bound(s); // expected-warning {{bound to num}}
}

// Free function with annotated param and ptr return
int* boo(int* num [[clang::lifetimebound]]) { return num; }

void caller_five() {
  int n = 55;
  int* n_ptr = &n;
  int* s = boo(n_ptr);

  clang_analyzer_lifetime_bound(s); // expected-warning {{bound to n}}
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

  clang_analyzer_lifetime_bound(bind); // expected-warning {{contains loan y}}
}

// Function returns a reference and has an annotated parameter
int& func(int& some_number [[clang::lifetimebound]]);

void caller_eight() {
  int f = 15;
  auto& bind = func(f);

  clang_analyzer_lifetime_bound(bind); // expected-warning {{contains loan f}}
}

// Function returns a reference and has two annotated parameters.
int& f(int& a [[clang::lifetimebound]], int& b [[clang::lifetimebound]]);

void caller_nine() {
  int first_num = 1;
  int second_num = 2;
  int& numbers = f(first_num, second_num);

  clang_analyzer_lifetime_bound(numbers); // expected-warning {{contains loan first_num}}

// FIXME: Currently the callback only iterates until the first annotated parameter which
// means the second annotation never gets read here. That is a clear bug. It should be fixed
// in order to analyze all the parameters which are annotated.
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
