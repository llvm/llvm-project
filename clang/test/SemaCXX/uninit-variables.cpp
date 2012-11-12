// RUN: %clang_cc1 -fsyntax-only -Wuninitialized -fsyntax-only -fcxx-exceptions %s -verify

// Stub out types for 'typeid' to work.
namespace std { class type_info {}; }

int test1_aux(int &x);
int test1() {
  int x;
  test1_aux(x);
  return x; // no-warning
}

int test2_aux() {
  int x;
  int &y = x;
  return x; // no-warning
}

// Don't warn on unevaluated contexts.
void unevaluated_tests() {
  int x;
  (void)sizeof(x);
  (void)typeid(x);
}

// Warn for glvalue arguments to typeid whose type is polymorphic.
struct A { virtual ~A() {} };
void polymorphic_test() {
  A *a; // expected-note{{initialize the variable 'a' to silence this warning}}
  (void)typeid(*a); // expected-warning{{variable 'a' is uninitialized when used here}}
}

// Handle cases where the CFG may constant fold some branches, thus
// mitigating the need for some path-sensitivity in the analysis.
unsigned test3_aux();
unsigned test3() {
  unsigned x = 0;
  const bool flag = true;
  if (flag && (x = test3_aux()) == 0) {
    return x;
  }
  return x;
}
unsigned test3_b() {
  unsigned x ;
  const bool flag = true;
  if (flag && (x = test3_aux()) == 0) {
    x = 1;
  }
  return x; // no-warning
}
unsigned test3_c() {
  unsigned x; // expected-note{{initialize the variable 'x' to silence this warning}}
  const bool flag = false;
  if (flag && (x = test3_aux()) == 0) {
    x = 1;
  }
  return x; // expected-warning{{variable 'x' is uninitialized when used here}}
}

enum test4_A {
 test4_A_a, test_4_A_b
};
test4_A test4() {
 test4_A a; // expected-note{{variable 'a' is declared here}}
 return a; // expected-warning{{variable 'a' is uninitialized when used here}}
}

// Test variables getting invalidated by function calls with reference arguments
// *AND* there are multiple invalidated arguments.
void test5_aux(int &, int &);

int test5() {
  int x, y;
  test5_aux(x, y);
  return x + y; // no-warning
}

// This test previously crashed Sema.
class Rdar9188004A {
public: 
  virtual ~Rdar9188004A();
};

template< typename T > class Rdar9188004B : public Rdar9188004A {
virtual double *foo(Rdar9188004B *next) const  {
    double *values = next->foo(0);
    try {
    }
    catch(double e) {
      values[0] = e;
    }
    return 0;
  }
};
class Rdar9188004C : public Rdar9188004B<Rdar9188004A> {
  virtual void bar(void) const;
};
void Rdar9188004C::bar(void) const {}

// Don't warn about uninitialized variables in unreachable code.
void PR9625() {
  if (false) {
    int x;
    (void)static_cast<float>(x); // no-warning
  }
}

// Don't warn about variables declared in "catch"
void RDar9251392_bar(const char *msg);

void RDar9251392() {
  try {
    throw "hi";
  }
  catch (const char* msg) {
    RDar9251392_bar(msg); // no-warning
  }
}

// Test handling of "no-op" casts.
void test_noop_cast()
{
    int x = 1;
    int y = (int&)x; // no-warning
}

void test_noop_cast2() {
    int x; // expected-note {{initialize the variable 'x' to silence this warning}}
    int y = (int&)x; // expected-warning {{uninitialized when used here}}
}

// Test handling of bit casts.
void test_bitcasts() {
  int x = 1;
  int y = (float &)x; // no-warning
}

void test_bitcasts_2() {
  int x;  // expected-note {{initialize the variable 'x' to silence this warning}}
  int y = (float &)x; // expected-warning {{uninitialized when used here}}
}

void consume_const_ref(const int &n);
int test_const_ref() {
  int n; // expected-note {{variable}}
  consume_const_ref(n);
  return n; // expected-warning {{uninitialized when used here}}
}
