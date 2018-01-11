// RUN: %clang_cc1 -std=c++1z -verify %s

class Bar {
  int myVal;
public:
  Bar();

  int getValue();
};

class Foo {
  Bar myBar;
public:
  Foo();

  Bar &getBar();
};

Bar &getBarFromFoo(Foo &f);
int getintFromBar(Bar &b);

int bar(int);
int foo(int);

struct PairT {
  int t1;
  int t2;
};

struct PairT baz(int);

int spawn_tests(int n) {
  Foo f;
  // n is evaluated before spawn, result is passsed by value.
  _Cilk_spawn bar(n);
  // f.getBar() is evalauted before spawn, result passed by value.  Only
  // getValue() call is spawned.
  _Cilk_spawn f.getBar().getValue();
  // getBarFromFoo(f) is evaluated before spawn.  Only getintFromBar() is
  // spawned.
  _Cilk_spawn getintFromBar(getBarFromFoo(f));
  // [&]{ getintFromBar(getBarFromFoo(f)); }();
  return 0;
}

int basic_spawn_assign_tests(int n, int *p) {
  Foo f;
  int x;
  // Call to bar and store to address of x are both spawned.
  x = _Cilk_spawn bar(n);
  // Address computation of *p happens before detach.
  *p = _Cilk_spawn bar(n);
  // Only the call to getValue() is spawned.
  int x1;
  x1 = _Cilk_spawn f.getBar().getValue();
  // Call to getBar() and subsequent copy is spawned.
  Bar fb;
  fb = _Cilk_spawn f.getBar();
  // Memory allocation and call to constructor are both spawned.  EH structures
  // are local to detached block, and detach-local EH terminates in a resume.
  Foo *myfoo;
  myfoo = _Cilk_spawn new Foo();
  struct PairT pair;
  pair = _Cilk_spawn baz(n-2);
  return 0;
}

int basic_spawn_decl_tests(int n) {
  Foo f;
  // Call to foo and store to address of y are spawned.
  int y = _Cilk_spawn foo(n);
  // Call to getValue and store to address of z are spawned.
  int z = _Cilk_spawn f.getBar().getValue();
  // Call to getBar() and subsequent copy is spawned.
  Bar fb  = _Cilk_spawn f.getBar();
  // Call to foo and store to a is spawned.  Call to bar and store to c is
  // spawned.
  int a = _Cilk_spawn foo(n-1), b = 7, c = _Cilk_spawn bar(n-1);
  // Call to foo, explicit cast, and store to yl are spawned.
  long yl = _Cilk_spawn (long)foo(n);
  // Call to foo, implicit cast, and store to yl2 are spawned.
  long yl2 = _Cilk_spawn foo(n);
  // Memory allocation, call to Bar() constructor, and store are spawned.
  Bar *mybar = _Cilk_spawn new Bar();
  // Call to baz and store are spawned.
  struct PairT pair = _Cilk_spawn baz(n-2);
  return 0;
}

int spawn_assign_eval_order_tests(int n) {
  int i = 0;
  int Arr[5];
  Arr[i++] = _Cilk_spawn bar(i++); // expected-warning {{multiple unsequenced modifications to 'i'}}
  Arr[i++] += bar(i); // expected-warning {{unsequenced modification and access to 'i'}}
  Arr[i++] += _Cilk_spawn bar(i); // expected-warning {{unsequenced modification and access to 'i'}}
  return 0;
}

