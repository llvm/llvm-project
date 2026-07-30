// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.cplusplus.UseAfterLifetimeEnd,debug.DebugLifetimeModeling \
// RUN:   -analyzer-config cfg-lifetime=true -analyzer-output=text -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.cplusplus.UseAfterLifetimeEnd,debug.DebugLifetimeModeling \
// RUN:   -analyzer-config c++-container-inlining=false -analyzer-config cfg-lifetime=true -analyzer-output=text -verify %s
struct A {};

struct Pair {
  int a;
  int b;
};

struct Base {
  int x;
};

struct Derived : Base {
  int y;
};

struct Inner {
  int val;
};

struct Outer {
  Inner inner;
};

struct Buffer {
  int arr[4];
};

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
  clang_analyzer_dumpLifetimeOriginsOf(r);
  // expected-warning@-1 {{Origin '&v' bound to 'v'}}
  // expected-note@-2    {{Origin '&v' bound to 'v'}}
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
  clang_analyzer_dumpLifetimeOriginsOf(f);
  // expected-warning@-1 {{Origin '&y.a' bound to 'y'}}
  // expected-note@-2    {{Origin '&y.a' bound to 'y'}}
}

// Obj ptr type function return annotated case.
struct Z {
  A a;
  A *getA() [[clang::lifetimebound]] { return &a; }
};

void caller_three() {
  Z z;
  A *func = z.getA();
  clang_analyzer_dumpLifetimeOriginsOf(func);
  // expected-warning@-1 {{Origin '&z.a' bound to 'z'}}
  // expected-note@-2    {{Origin '&z.a' bound to 'z'}}
}

// Free function with annotated param and ref return.
int &foo(int &num [[clang::lifetimebound]]) { return num; }

void caller_four() {
  int num = 5;
  int &s = foo(num);
  clang_analyzer_dumpLifetimeOriginsOf(s);
  // expected-warning@-1 {{Origin '&num' bound to 'num'}}
  // expected-note@-2    {{Origin '&num' bound to 'num'}}
}

// Free function with annotated param and ptr return.
int *boo(int *num [[clang::lifetimebound]]) { return num; }

void caller_five() {
  int n = 55;
  int *n_ptr = &n;
  int *s = boo(n_ptr);

  clang_analyzer_dumpLifetimeOriginsOf(s);
  // expected-warning@-1 {{Origin '&n' bound to 'n'}}
  // expected-note@-2  {{Origin '&n' bound to 'n'}}
}

// Free function with both annotated and non-annotated parameters.
int &fn(int &f, int &s [[clang::lifetimebound]]) { return s; }

void caller_six() {
  int even = 50;
  int odd = 55;
  int &s = fn(even, odd);

  clang_analyzer_dumpLifetimeOriginsOf(s);
  // expected-warning@-1 {{Origin '&odd' bound to 'odd'}}
  // expected-note@-2    {{Origin '&odd' bound to 'odd'}}
}

// Test cases for testing when the result of function calls are SymbolRefs.

// Function returns ptr and has an annotated parameter.
int *foo(int *n [[clang::lifetimebound]]);

void caller_seven() {
  int y = 15;
  int *y_ptr = &y;
  auto *bind = foo(y_ptr);

  clang_analyzer_dumpLifetimeOriginsOf(bind);
  // expected-warning-re@-1 {{Origin '&SymRegion{{.*}}' bound to 'y'}}
  // expected-note-re@-2    {{Origin '&SymRegion{{.*}}' bound to 'y'}}
}

// Function returns a reference and has an annotated parameter.
int &func(int &some_number [[clang::lifetimebound]]);

void caller_eight() {
  int f = 15;
  auto &bind = func(f);

  clang_analyzer_dumpLifetimeOriginsOf(bind);
  // expected-warning-re@-1 {{Origin '&SymRegion{{.*}}' bound to 'f'}}
  // expected-note-re@-2    {{Origin '&SymRegion{{.*}}' bound to 'f'}}
}

// Function returns a reference and has two annotated parameters.
int &f(int &a [[clang::lifetimebound]], int &b [[clang::lifetimebound]]);

void caller_nine() {
  int first_num = 1;
  int second_num = 2;
  int &numbers = f(first_num, second_num);

  clang_analyzer_dumpLifetimeOriginsOf(numbers);
  // expected-warning-re@-1 {{Origin '&SymRegion{{.*}}' bound to 'first_num', 'second_num'}}
  // expected-note-re@-2    {{Origin '&SymRegion{{.*}}' bound to 'first_num', 'second_num'}}
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
  int i = 5; //expected-note {{'i' initialized here}}
  return test_func(&i);
  // expected-warning@-1 {{Returning value bound to 'i' that will go out of scope}}
  // expected-warning@-2 {{address of stack memory associated with local variable 'i' returned}}
  // expected-note@-3    {{Returning value bound to 'i' that will go out of scope}}
}

int *variable_return() {
  int y = 5; // expected-note {{'y' initialized here}}
  int *p = test_func(&y);
  return p;
  // expected-warning@-1 {{Returning value bound to 'y' that will go out of scope}}
  // expected-note@-2    {{Returning value bound to 'y' that will go out of scope}}
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
  // expected-note@-1 {{'x' initialized here}}
  // expected-note@-2 {{'y' initialized here}}
  return multi_param_test_ref(x, y);
  // expected-warning@-1 {{Returning value bound to 'x' that will go out of scope}}
  // expected-note@-2    {{Returning value bound to 'x' that will go out of scope}}
  // expected-warning@-3 {{Returning value bound to 'y' that will go out of scope}}
  // expected-note@-4    {{Returning value bound to 'y' that will go out of scope}}
  // expected-warning@-5 {{reference to stack memory associated with local variable 'x' returned}}
  // expected-warning@-6 {{reference to stack memory associated with local variable 'y' returned}}
}

// Return value bound to annotated parameters (no dangling sources).
int &no_dangling_sources_ref(int &a [[clang::lifetimebound]], int &b [[clang::lifetimebound]]) {
  return multi_param_test_ref(a, b); // no-warning
}

// Return value bound to annotated parameters (one dangling source).
int &one_dangling_source_ref(int &a [[clang::lifetimebound]]) {
  int x = 1; // expected-note {{'x' initialized here}}
  return multi_param_test_ref(a, x);
  // expected-warning@-1 {{Returning value bound to 'x' that will go out of scope}}
  // expected-note@-2    {{Returning value bound to 'x' that will go out of scope}}
  // expected-warning@-3 {{reference to stack memory associated with local variable 'x' returned}}
}

int *multi_param_test_ptr(int *a [[clang::lifetimebound]], int *b [[clang::lifetimebound]]);

// Return value bound to annotated parameters (two dangling sources).
int *dangling_sources_ptr() {
  int x = 1, y = 2;
  // expected-note@-1 {{'x' initialized here}}
  // expected-note@-2 {{'y' initialized here}}
  int *x_ptr = &x;
  int *y_ptr = &y;
  return multi_param_test_ptr(x_ptr, y_ptr);
  // expected-warning@-1 {{Returning value bound to 'x' that will go out of scope}}
  // expected-note@-2    {{Returning value bound to 'x' that will go out of scope}}
  // expected-warning@-3 {{Returning value bound to 'y' that will go out of scope}}
  // expected-note@-4    {{Returning value bound to 'y' that will go out of scope}}
}

// Return value bound to annotated parameters (no dangling sources).
int *no_dangling_sources_ptr(int *a [[clang::lifetimebound]], int *b [[clang::lifetimebound]]) {
  return multi_param_test_ptr(a, b); // no-warning
}

// Return value bound to annotated parameters (one dangling source).
int *one_dangling_source_ptr(int *a [[clang::lifetimebound]]) {
  int x = 1; // expected-note {{'x' initialized here}} 
  int *x_ptr = &x;
  return multi_param_test_ptr(a, x_ptr);
  // expected-warning@-1 {{Returning value bound to 'x' that will go out of scope}}
  // expected-note@-2    {{Returning value bound to 'x' that will go out of scope}}
}

struct S {
    int x;
    int* get() [[clang::lifetimebound]] { return &x; }
    S() = default;
    S(S&& other) { (void)other.get(); }
    ~S() { get(); }
};

S make() { return S(); }

S passThrough(S param) { return param; }

void outer() {
    auto f = passThrough(make());
    (void)f; // no-warning
}

int *danglingLocal() {
  S s; // expected-note {{'s' initialized here}}
  return s.get(); // expected-note {{Returning value bound to 's' that will go out of scope}}
  // expected-warning@-1 {{Returning value bound to 's' that will go out of scope}}
  // expected-warning@-2 {{Address of stack memory associated with local variable 's' returned}}
  // expected-note@-3    {{Address of stack memory associated with local variable 's' returned to caller}}
  // expected-warning@-4 {{address of stack memory associated with local variable 's' returned}}
}

int *danglingParam(S param) {
  return param.get();
  // expected-warning@-1 {{Returning value bound to 'param' that will go out of scope}}
  // expected-note@-2    {{Returning value bound to 'param' that will go out of scope}}
  // expected-warning@-3 {{Address of stack memory associated with local variable 'param' returned}}
  // expected-note@-4    {{Address of stack memory associated with local variable 'param' returned to caller}}
  // expected-warning@-5 {{address of stack memory associated with parameter 'param' returned}}
}

int *getFieldPtr(Pair &p [[clang::lifetimebound]]) { return &p.a; }

int *field_subobject_dangling() {
  Pair pair{3, 5}; // expected-note {{'pair' initialized here}}
  return getFieldPtr(pair);
  // expected-warning@-1 {{Returning value bound to 'pair' that will go out of scope}}
  // expected-note@-2    {{Returning value bound to 'pair' that will go out of scope}}
  // expected-warning@-3 {{Address of stack memory associated with local variable 'pair' returned to caller}}
  // expected-note@-4    {{Address of stack memory associated with local variable 'pair' returned to caller}}
  // expected-warning@-5 {{address of stack memory associated with local variable 'pair' returned}}
}

int *getBasePtr(Derived &d [[clang::lifetimebound]]) {
  return &static_cast<Base &>(d).x;
}

int *base_subobject_dangling() {
  Derived derived{}; // expected-note {{'derived' initialized here}}
  return getBasePtr(derived);
  // expected-warning@-1 {{Returning value bound to 'derived' that will go out of scope}}
  // expected-note@-2    {{Returning value bound to 'derived' that will go out of scope}}
  // expected-warning@-3 {{Address of stack memory associated with local variable 'derived' returned to caller}}
  // expected-note@-4    {{Address of stack memory associated with local variable 'derived' returned to caller}}
  // expected-warning@-5 {{address of stack memory associated with local variable 'derived' returned}}
}

int *getNestedFieldPtr(Outer &o [[clang::lifetimebound]]) {
  return &o.inner.val;
}

int *nested_subobject_dangling() {
  Outer outer{}; // expected-note {{'outer' initialized here}}
  return getNestedFieldPtr(outer);
  // expected-warning@-1 {{Returning value bound to 'outer' that will go out of scope}}
  // expected-note@-2    {{Returning value bound to 'outer' that will go out of scope}} 
  // expected-warning@-3 {{Address of stack memory associated with local variable 'outer' returned to caller}}
  // expected-note@-4    {{Address of stack memory associated with local variable 'outer' returned to caller}}
  // expected-warning@-5 {{address of stack memory associated with local variable 'outer' returned}}
}

int *getArrayElementPtr(Buffer &b [[clang::lifetimebound]]) {
  return &b.arr[0];
}

int *array_member_subobject_dangling() {
  Buffer buf{}; // expected-note {{'buf' initialized here}}
  return getArrayElementPtr(buf);
  // expected-warning@-1 {{Returning value bound to 'buf' that will go out of scope}}
  // expected-note@-2    {{Returning value bound to 'buf' that will go out of scope}}
  // expected-warning@-3 {{Address of stack memory associated with local variable 'buf' returned to caller}}
  // expected-note@-4    {{Address of stack memory associated with local variable 'buf' returned to caller}}
  // expected-warning@-5 {{address of stack memory associated with local variable 'buf' returned}}
}
