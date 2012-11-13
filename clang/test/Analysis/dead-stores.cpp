// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -analyze -analyzer-checker=deadcode.DeadStores -verify -Wno-unreachable-code %s
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -analyze -analyzer-store=region -analyzer-constraints=range -analyzer-checker=deadcode.DeadStores -verify -Wno-unreachable-code %s

//===----------------------------------------------------------------------===//
// Basic dead store checking (but in C++ mode).
//===----------------------------------------------------------------------===//

int j;
void test1() {
  int x = 4;

  x = x + 1; // expected-warning{{never read}}

  switch (j) {
  case 1:
    throw 1;
    (void)x;
    break;
  }
}

//===----------------------------------------------------------------------===//
// Dead store checking involving constructors.
//===----------------------------------------------------------------------===//

class Test2 {
  int &x;
public:
  Test2(int &y) : x(y) {}
  ~Test2() { ++x; }
};

int test2(int x) {
  { Test2 a(x); } // no-warning
  return x;
}

//===----------------------------------------------------------------------===//
// Dead store checking involving CXXTemporaryExprs
//===----------------------------------------------------------------------===//

namespace TestTemp {
  template<typename _Tp>
  class pencil {
  public:
    ~pencil() throw() {}
  };
  template<typename _Tp, typename _Number2> struct _Row_base {
    _Row_base(const pencil<_Tp>& x) {}
  };
  template<typename _Tp, typename _Number2 = TestTemp::pencil<_Tp> >
  class row : protected _Row_base<_Tp, _Number2>     {
    typedef _Row_base<_Tp, _Number2> _Base;
    typedef _Number2 pencil_type;
  public:
    explicit row(const pencil_type& __a = pencil_type()) : _Base(__a) {}
  };
}

void test2_b() {
  TestTemp::row<const char*> x; // no-warning
}

//===----------------------------------------------------------------------===//
// Test references.
//===----------------------------------------------------------------------===//

void test3_a(int x) {
   x = x + 1; // expected-warning{{never read}}
}

void test3_b(int &x) {
  x = x + 1; // no-warninge
}

void test3_c(int x) {
  int &y = x;
  // Shows the limitation of dead stores tracking.  The write is really
  // dead since the value cannot escape the function.
  ++y; // no-warning
}

void test3_d(int &x) {
  int &y = x;
  ++y; // no-warning
}

void test3_e(int &x) {
  int &y = x;
}

//===----------------------------------------------------------------------===//
// Dead stores involving 'new'
//===----------------------------------------------------------------------===//

static void test_new(unsigned n) {
  char **p = new char* [n]; // expected-warning{{never read}}
}

//===----------------------------------------------------------------------===//
// Dead stores in namespaces.
//===----------------------------------------------------------------------===//

namespace foo {
  int test_4(int x) {
    x = 2; // expected-warning{{Value stored to 'x' is never read}}
    x = 2;
    return x;
  }
}

//===----------------------------------------------------------------------===//
// Dead stores in with EH code.
//===----------------------------------------------------------------------===//

void test_5_Aux();
int test_5() {
  int x = 0;
  try {
    x = 2; // no-warning
    test_5_Aux();
  }
  catch (int z) {
    return x + z;
  }
  return 1;
}


int test_6_aux(unsigned x);

void test_6() {
  unsigned currDestLen = 0;  // no-warning
  try {
    while (test_6_aux(currDestLen)) {
      currDestLen += 2; // no-warning
    } 
  }
  catch (void *) {}
}

void test_6b() {
  unsigned currDestLen = 0;  // no-warning
  try {
    while (test_6_aux(currDestLen)) {
      currDestLen += 2; // expected-warning {{Value stored to 'currDestLen' is never read}}
      break;
    } 
  }
  catch (void *) {}
}
