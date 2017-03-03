// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus.NewDeleteLeaks -verify %s

class A0 {};

class A1 {
public:
  A1(int);
};

struct S{
  int i;
};

class A2 {
public:
  A2();
  A2(S);
  A2(int*);
  A2(S*);
  A2(S&, int);
  A2(int, S**);
};

void test() {
  new int; // expected-warning@+1 {{Potential memory leak}}
  new A0; // expected-warning@+1 {{Potential memory leak}}
  new A1(0); // expected-warning@+1 {{Potential memory leak}}
  new A2; // expected-warning@+1 {{Potential memory leak}}
  S s;
  s.i = 1;
  S* ps = new S;
  new A2(s); // expected-warning@+1 {{Potential memory leak}}
  new A2(&(s.i)); // expected-warning@+1 {{Potential memory leak}}
  new A2(ps); // no warning
  new A2(*ps, 1); // no warning
  new A2(1, &ps); // no warning

  // Tests to ensure that leaks are reported for consumed news no matter what the arguments are.
  A2 *a2p1 = new A2; // expected-warning@+1 {{Potential leak of memory}}
  A2 *a2p2 = new A2(ps); // expected-warning@+1 {{Potential leak of memory}}
  A2 *a2p3 = new A2(*ps, 1); // expected-warning@+1 {{Potential leak of memory}}
  A2 *a2p4 = new A2(1, &ps); // expected-warning@+1 {{Potential leak of memory}}
}
