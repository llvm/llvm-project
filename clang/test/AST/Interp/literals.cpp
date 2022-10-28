// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++11 -verify %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++20 -verify %s
// RUN: %clang_cc1 -std=c++11 -verify=ref %s
// RUN: %clang_cc1 -std=c++20 -verify=ref %s

#define INT_MIN (~__INT_MAX__)
#define INT_MAX __INT_MAX__


static_assert(true, "");
static_assert(false, ""); // expected-error{{failed}} ref-error{{failed}}
static_assert(nullptr == nullptr, "");
static_assert(1 == 1, "");
static_assert(1 == 3, ""); // expected-error{{failed}} ref-error{{failed}}

constexpr int number = 10;
static_assert(number == 10, "");
static_assert(number != 10, ""); // expected-error{{failed}} \
                                 // ref-error{{failed}} \
                                 // expected-note{{evaluates to}} \
                                 // ref-note{{evaluates to}}

constexpr bool b = number;
static_assert(b, "");
constexpr int one = true;
static_assert(one == 1, "");

namespace IntegralCasts {
  constexpr int i = 12;
  constexpr unsigned int ui = i;
  static_assert(ui == 12, "");
  constexpr unsigned int ub = !false;
  static_assert(ub == 1, "");

  constexpr int si = ui;
  static_assert(si == 12, "");
  constexpr int sb = true;
  static_assert(sb == 1, "");

  constexpr int zero = 0;
  constexpr unsigned int uzero = 0;
  constexpr bool bs = i;
  static_assert(bs, "");
  constexpr bool bu = ui;
  static_assert(bu, "");
  constexpr bool ns = zero;
  static_assert(!ns, "");
  constexpr bool nu = uzero;
  static_assert(!nu, "");
};



constexpr bool getTrue() { return true; }
constexpr bool getFalse() { return false; }
constexpr void* getNull() { return nullptr; }

constexpr int neg(int m) { return -m; }
constexpr bool inv(bool b) { return !b; }

static_assert(12, "");
static_assert(12 == -(-(12)), "");
static_assert(!false, "");
static_assert(!!true, "");
static_assert(!!true == !false, "");
static_assert(true == 1, "");
static_assert(false == 0, "");
static_assert(!5 == false, "");
static_assert(!0, "");
static_assert(-true, "");
static_assert(-false, ""); //expected-error{{failed}} ref-error{{failed}}

static_assert(~0 == -1, "");
static_assert(~1 == -2, "");
static_assert(~-1 == 0, "");
static_assert(~255 == -256, "");
static_assert(~INT_MIN == INT_MAX, "");
static_assert(~INT_MAX == INT_MIN, "");

enum E {};
constexpr E e = static_cast<E>(0);
static_assert(~e == -1, "");


constexpr int m = 10;
constexpr const int *p = &m;
static_assert(p != nullptr, "");
static_assert(*p == 10, "");

constexpr const int* getIntPointer() {
  return &m;
}
static_assert(getIntPointer() == &m, "");
static_assert(*getIntPointer() == 10, "");

constexpr int gimme(int k) {
  return k;
}
static_assert(gimme(5) == 5, "");

namespace SizeOf {
  constexpr int soint = sizeof(int);
  constexpr int souint = sizeof(unsigned int);
  static_assert(soint == souint, "");

  static_assert(sizeof(&soint) == sizeof(void*), "");
  static_assert(sizeof(&soint) == sizeof(nullptr), "");

  static_assert(sizeof(long) == sizeof(unsigned long), "");
  static_assert(sizeof(char) == sizeof(unsigned char), "");

  constexpr int N = 4;
  constexpr int arr[N] = {1,2,3,4};
  static_assert(sizeof(arr) == N * sizeof(int), "");
  static_assert(sizeof(arr) == N * sizeof(arr[0]), "");

  constexpr bool arrB[N] = {true, true, true, true};
  static_assert(sizeof(arrB) == N * sizeof(bool), "");

  static_assert(sizeof(bool) == 1, "");
  static_assert(sizeof(char) == 1, "");

  constexpr int F = sizeof(void); // expected-error{{incomplete type 'void'}} \
                                  // ref-error{{incomplete type 'void'}}

  constexpr int F2 = sizeof(gimme); // expected-error{{to a function type}} \
                                    // ref-error{{to a function type}}



  /// FIXME: The following code should be accepted.
  struct S {
    void func();
  };
  constexpr void (S::*Func)() = &S::func; // expected-error {{must be initialized by a constant expression}} \
                                          // expected-error {{interpreter failed to evaluate an expression}}
  static_assert(sizeof(Func) == sizeof(&S::func), "");


  void func() {
    int n = 12;
    constexpr int oofda = sizeof(int[n++]); // expected-error {{must be initialized by a constant expression}} \
                                            // ref-error {{must be initialized by a constant expression}}
  }


#if __cplusplus >= 202002L
  /// FIXME: The following code should be accepted.
  consteval int foo(int n) { // ref-error {{consteval function never produces a constant expression}}
    return sizeof(int[n]); // ref-note 3{{not valid in a constant expression}} \
                           // expected-note {{not valid in a constant expression}}
  }
  constinit int var = foo(5); // ref-error {{not a constant expression}} \
                              // ref-note 2{{in call to}} \
                              // ref-error {{does not have a constant initializer}} \
                              // ref-note {{required by 'constinit' specifier}} \
                              // expected-error  {{is not a constant expression}} \
                              // expected-note {{in call to}} \
                              // expected-error {{does not have a constant initializer}} \
                              // expected-note {{required by 'constinit' specifier}} \

#endif
};

namespace rem {
  static_assert(2 % 2 == 0, "");
  static_assert(2 % 1 == 0, "");
  static_assert(-3 % 4 == -3, "");
  static_assert(4 % -2 == 0, "");
  static_assert(-3 % -4 == -3, "");

  constexpr int zero() { return 0; }
  static_assert(10 % zero() == 20, ""); // ref-error {{not an integral constant expression}} \
                                        // ref-note {{division by zero}} \
                                        // expected-error {{not an integral constant expression}} \
                                        // expected-note {{division by zero}}


  static_assert(true % true == 0, "");
  static_assert(false % true == 0, "");
  static_assert(true % false == 10, ""); // ref-error {{not an integral constant expression}} \
                                         // ref-note {{division by zero}} \
                                         // expected-error {{not an integral constant expression}} \
                                         // expected-note {{division by zero}}
  constexpr int x = INT_MIN % - 1; // ref-error {{must be initialized by a constant expression}} \
                                   // ref-note {{value 2147483648 is outside the range}} \
                                   // expected-error {{must be initialized by a constant expression}} \
                                   // expected-note {{value 2147483648 is outside the range}} \

};

namespace div {
  constexpr int zero() { return 0; }
  static_assert(12 / 3 == 4, "");
  static_assert(12 / zero() == 12, ""); // ref-error {{not an integral constant expression}} \
                                        // ref-note {{division by zero}} \
                                        // expected-error {{not an integral constant expression}} \
                                        // expected-note {{division by zero}}
  static_assert(12 / -3 == -4, "");
  static_assert(-12 / 3 == -4, "");


  constexpr int LHS = 12;
  constexpr long unsigned RHS = 3;
  static_assert(LHS / RHS == 4, "");

  constexpr int x = INT_MIN / - 1; // ref-error {{must be initialized by a constant expression}} \
                                   // ref-note {{value 2147483648 is outside the range}} \
                                   // expected-error {{must be initialized by a constant expression}} \
                                   // expected-note {{value 2147483648 is outside the range}} \

};

namespace cond {
  constexpr bool isEven(int n) {
    return n % 2 == 0 ? true : false;
  }
  static_assert(isEven(2), "");
  static_assert(!isEven(3), "");
  static_assert(isEven(100), "");

  constexpr int M = 5 ? 10 : 20;
  static_assert(M == 10, "");

  static_assert(5 ? 13 : 16 == 13, "");
  static_assert(0 ? 13 : 16 == 16, "");

  static_assert(number ?: -15 == number, "");
  static_assert(0 ?: 100 == 100 , "");

#if __cplusplus >= 201402L
  constexpr int N = 20;
  constexpr int foo() {
    int m = N > 0 ? 5 : 10;

    return m == 5 ? isEven(m) : true;
  }
  static_assert(foo() == false, "");

  constexpr int dontCallMe(unsigned m) {
    if (m == 0) return 0;
    return dontCallMe(m - 2);
  }

  // Can't call this because it will run into infinite recursion.
  constexpr int assertNotReached() {
    return dontCallMe(3);
  }

  constexpr int testCond() {
    return true ? 5 : assertNotReached();
  }

  constexpr int testCond2() {
    return false ? assertNotReached() : 10;
  }

  static_assert(testCond() == 5, "");
  static_assert(testCond2() == 10, "");

#endif

};

namespace band {
  static_assert((10 & 1) == 0, "");
  static_assert((10 & 10) == 10, "");

  static_assert((1337 & -1) == 1337, "");
  static_assert((0 & gimme(12)) == 0, "");
};

namespace bitOr {
  static_assert((10 | 1) == 11, "");
  static_assert((10 | 10) == 10, "");

  static_assert((1337 | -1) == -1, "");
  static_assert((0 | gimme(12)) == 12, "");
  static_assert((12 | true) == 13, "");
};
