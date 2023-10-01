// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -fms-extensions -std=c++11 -verify %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -fms-extensions -std=c++20 -verify %s
// RUN: %clang_cc1 -std=c++11 -fms-extensions -verify=ref %s
// RUN: %clang_cc1 -std=c++20 -fms-extensions -verify=ref %s

#define INT_MIN (~__INT_MAX__)
#define INT_MAX __INT_MAX__

typedef __INTPTR_TYPE__ intptr_t;


static_assert(true, "");
static_assert(false, ""); // expected-error{{failed}} ref-error{{failed}}
static_assert(nullptr == nullptr, "");
static_assert(__null == __null, "");
static_assert(1 == 1, "");
static_assert(1 == 3, ""); // expected-error{{failed}} ref-error{{failed}}

constexpr void* v = nullptr;
static_assert(__null == v, "");

constexpr int number = 10;
static_assert(number == 10, "");
static_assert(number != 10, ""); // expected-error{{failed}} \
                                 // ref-error{{failed}} \
                                 // expected-note{{evaluates to}} \
                                 // ref-note{{evaluates to}}


#ifdef __SIZEOF__INT128__
namespace i128 {
  typedef __int128 int128_t;
  typedef unsigned __int128 uint128_t;
  constexpr int128_t I128_1 = 12;
  static_assert(I128_1 == 12, "");
  static_assert(I128_1 != 10, "");
  static_assert(I128_1 != 12, ""); // expected-error{{failed}} \
                                   // ref-error{{failed}} \
                                   // expected-note{{evaluates to}} \
                                   // ref-note{{evaluates to}}

  static const __uint128_t UINT128_MAX =__uint128_t(__int128_t(-1L));
  static_assert(UINT128_MAX == -1, "");

  static const __int128_t INT128_MAX = UINT128_MAX >> (__int128_t)1;
  static_assert(INT128_MAX != 0, "");
  static const __int128_t INT128_MIN = -INT128_MAX - 1;
  constexpr __int128 A = INT128_MAX + 1; // expected-error {{must be initialized by a constant expression}} \
                                         // expected-note {{outside the range}} \
                                         // ref-error {{must be initialized by a constant expression}} \
                                         // ref-note {{outside the range}}
  constexpr int128_t Two = (int128_t)1 << 1ul;
  static_assert(Two == 2, "");

#if __cplusplus >= 201402L
  template <typename T>
  constexpr T CastFrom(__int128_t A) {
    T B = (T)A;
    return B;
  }
  static_assert(CastFrom<char>(12) == 12, "");
  static_assert(CastFrom<unsigned char>(12) == 12, "");
  static_assert(CastFrom<long>(12) == 12, "");
  static_assert(CastFrom<unsigned short>(12) == 12, "");
  static_assert(CastFrom<int128_t>(12) == 12, "");
  static_assert(CastFrom<float>(12) == 12, "");
  static_assert(CastFrom<double>(12) == 12, "");
  static_assert(CastFrom<long double>(12) == 12, "");

  template <typename T>
  constexpr __int128 CastTo(T A) {
    int128_t B = (int128_t)A;
    return B;
  }
  static_assert(CastTo<char>(12) == 12, "");
  static_assert(CastTo<unsigned char>(12) == 12, "");
  static_assert(CastTo<long>(12) == 12, "");
  static_assert(CastTo<unsigned long long>(12) == 12, "");
  static_assert(CastTo<float>(12) == 12, "");
  static_assert(CastTo<double>(12) == 12, "");
  static_assert(CastTo<long double>(12) == 12, "");
#endif

constexpr int128_t Error = __LDBL_MAX__; // ref-warning {{implicit conversion of out of range value}} \
                                         // ref-error {{must be initialized by a constant expression}} \
                                         // ref-note {{is outside the range of representable values of type}} \
                                         // expected-warning {{implicit conversion of out of range value}} \
                                         // expected-error {{must be initialized by a constant expression}} \
                                         // expected-note {{is outside the range of representable values of type}}
}
#endif

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

constexpr int UninitI; // expected-error {{must be initialized by a constant expression}} \
                       // ref-error {{must be initialized by a constant expression}}
constexpr int *UninitPtr; // expected-error {{must be initialized by a constant expression}} \
                          // ref-error {{must be initialized by a constant expression}}

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

static_assert(-(1 << 31), ""); // expected-error {{not an integral constant expression}} \
                               // expected-note {{outside the range of representable values}} \
                               // ref-error {{not an integral constant expression}} \
                               // ref-note {{outside the range of representable values}} \

namespace PrimitiveEmptyInitList {
  constexpr int a = {};
  static_assert(a == 0, "");
  constexpr bool b = {};
  static_assert(!b, "");
  constexpr double d = {};
  static_assert(d == 0.0, "");
}


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

namespace PointerToBool {

  constexpr void *N = nullptr;
  constexpr bool B = N;
  static_assert(!B, "");
  static_assert(!N, "");

  constexpr float F = 1.0;
  constexpr const float *FP = &F;
  static_assert(FP, "");
  static_assert(!!FP, "");
}

namespace PointerComparison {

  struct S { int a, b; } s;
  constexpr void *null = 0;
  constexpr void *pv = (void*)&s.a;
  constexpr void *qv = (void*)&s.b;
  constexpr bool v1 = null < (int*)0;
  constexpr bool v2 = null < pv; // expected-error {{must be initialized by a constant expression}} \
                                 // expected-note {{comparison between 'nullptr' and '&s.a' has unspecified value}} \
                                 // ref-error {{must be initialized by a constant expression}} \
                                 // ref-note {{comparison between 'nullptr' and '&s.a' has unspecified value}} \

  constexpr bool v3 = null == pv; // ok
  constexpr bool v4 = qv == pv; // ok

  /// FIXME: These two are rejected by the current interpreter, but
  ///   accepted by GCC.
  constexpr bool v5 = qv >= pv; // ref-error {{constant expression}} \
                                // ref-note {{unequal pointers to void}}
  constexpr bool v8 = qv > (void*)&s.a; // ref-error {{constant expression}} \
                                        // ref-note {{unequal pointers to void}}
  constexpr bool v6 = qv > null; // expected-error {{must be initialized by a constant expression}} \
                                 // expected-note {{comparison between '&s.b' and 'nullptr' has unspecified value}} \
                                 // ref-error {{must be initialized by a constant expression}} \
                                 // ref-note {{comparison between '&s.b' and 'nullptr' has unspecified value}}

  constexpr bool v7 = qv <= (void*)&s.b; // ok
}

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


  struct S {
    void func();
  };
  constexpr void (S::*Func)() = &S::func;
  static_assert(sizeof(Func) == sizeof(&S::func), "");


  void func() {
    int n = 12;
    constexpr int oofda = sizeof(int[n++]); // expected-error {{must be initialized by a constant expression}} \
                                            // ref-error {{must be initialized by a constant expression}}
  }

#if __cplusplus >= 201402L
  constexpr int IgnoredRejected() { // ref-error {{never produces a constant expression}}
    int n = 0;
    sizeof(int[n++]); // expected-warning {{expression result unused}} \
                      // ref-warning {{expression result unused}} \
                      // ref-note 2{{subexpression not valid in a constant expression}}
    return n;
  }
  /// FIXME: This is rejected because the parameter so sizeof() is not constant.
  ///   produce a proper diagnostic.
  static_assert(IgnoredRejected() == 0, ""); // expected-error {{not an integral constant expression}} \
                                             // ref-error {{not an integral constant expression}} \
                                             // ref-note {{in call to 'IgnoredRejected()'}}
#endif


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

namespace bitXor {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wxor-used-as-pow"
  static_assert((10 ^ 1) == 11, "");
  static_assert((10 ^ 10) == 0, "");

  enum {
    ONE = 1,
  };

  static_assert((1337 ^ -1) == -1338, "");
  static_assert((0 | gimme(12)) == 12, "");
  static_assert((12 ^ true) == 13, "");
  static_assert((12 ^ ONE) == 13, "");
#pragma clang diagnostic pop
};

#if __cplusplus >= 201402L
constexpr bool IgnoredUnary() {
  bool bo = true;
  !bo; // expected-warning {{expression result unused}} \
       // ref-warning {{expression result unused}}
  return bo;
}
static_assert(IgnoredUnary(), "");
#endif

namespace strings {
  constexpr const char *S = "abc";
  static_assert(S[0] == 97, "");
  static_assert(S[1] == 98, "");
  static_assert(S[2] == 99, "");
  static_assert(S[3] == 0, "");

  static_assert("foobar"[2] == 'o', "");
  static_assert(2["foobar"] == 'o', "");

  constexpr const wchar_t *wide = L"bar";
  static_assert(wide[0] == L'b', "");

  constexpr const char32_t *u32 = U"abc";
  static_assert(u32[1] == U'b', "");

  constexpr char32_t c = U'\U0001F60E';
  static_assert(c == 0x0001F60EL, "");

  constexpr char k = -1;
  static_assert(k == -1, "");

  static_assert('\N{LATIN CAPITAL LETTER E}' == 'E', "");
  static_assert('\t' == 9, "");

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmultichar"
  constexpr int mc = 'abc';
  static_assert(mc == 'abc', "");
  __WCHAR_TYPE__ wm = L'abc'; // ref-error{{wide character literals may not contain multiple characters}} \
                              // expected-error{{wide character literals may not contain multiple characters}}
  __WCHAR_TYPE__ wu = u'abc'; // ref-error{{Unicode character literals may not contain multiple characters}} \
                              // expected-error{{Unicode character literals may not contain multiple characters}}
  __WCHAR_TYPE__ wU = U'abc'; // ref-error{{Unicode character literals may not contain multiple characters}} \
                              // expected-error{{Unicode character literals may not contain multiple characters}}
#if __cplusplus > 201103L
  __WCHAR_TYPE__ wu8 = u8'abc'; // ref-error{{Unicode character literals may not contain multiple characters}} \
                                // expected-error{{Unicode character literals may not contain multiple characters}}
#endif

#pragma clang diagnostic pop

  constexpr char foo[12] = "abc";
  static_assert(foo[0] == 'a', "");
  static_assert(foo[1] == 'b', "");
  static_assert(foo[2] == 'c', "");
  static_assert(foo[3] == 0, "");
  static_assert(foo[11] == 0, "");

  constexpr char foo2[] = "abc\0def";
  static_assert(foo2[0] == 'a', "");
  static_assert(foo2[3] == '\0', "");
  static_assert(foo2[6] == 'f', "");
  static_assert(foo2[7] == '\0', "");
  static_assert(foo2[8] == '\0', ""); // expected-error {{not an integral constant expression}} \
                                      // expected-note {{read of dereferenced one-past-the-end pointer}} \
                                      // ref-error {{not an integral constant expression}} \
                                      // ref-note {{read of dereferenced one-past-the-end pointer}}

  constexpr char foo3[4] = "abc";
  static_assert(foo3[3] == '\0', "");
  static_assert(foo3[4] == '\0', ""); // expected-error {{not an integral constant expression}} \
                                      // expected-note {{read of dereferenced one-past-the-end pointer}} \
                                      // ref-error {{not an integral constant expression}} \
                                      // ref-note {{read of dereferenced one-past-the-end pointer}}

  constexpr char foo4[2] = "abcd"; // expected-error {{initializer-string for char array is too long}} \
                                   // ref-error {{initializer-string for char array is too long}}
  static_assert(foo4[0] == 'a', "");
  static_assert(foo4[1] == 'b', "");
  static_assert(foo4[2] == '\0', ""); // expected-error {{not an integral constant expression}} \
                                      // expected-note {{read of dereferenced one-past-the-end pointer}} \
                                      // ref-error {{not an integral constant expression}} \
                                      // ref-note {{read of dereferenced one-past-the-end pointer}}

constexpr char foo5[12] = "abc\xff";
#if defined(__CHAR_UNSIGNED__) || __CHAR_BIT__ > 8
static_assert(foo5[3] == 255, "");
#else
static_assert(foo5[3] == -1, "");
#endif
};

#if __cplusplus > 201402L
namespace IncDec {
  constexpr int zero() {
    int a = 0;
    a++;
    ++a;
    a--;
    --a;
    return a;
  }
  static_assert(zero() == 0, "");

  constexpr int preInc() {
    int a = 0;
    return ++a;
  }
  static_assert(preInc() == 1, "");

  constexpr int postInc() {
    int a = 0;
    return a++;
  }
  static_assert(postInc() == 0, "");

  constexpr int preDec() {
    int a = 0;
    return --a;
  }
  static_assert(preDec() == -1, "");

  constexpr int postDec() {
    int a = 0;
    return a--;
  }
  static_assert(postDec() == 0, "");

  constexpr int three() {
    int a = 0;
    return ++a + ++a; // expected-warning {{multiple unsequenced modifications to 'a'}} \
                      // ref-warning {{multiple unsequenced modifications to 'a'}} \

  }
  static_assert(three() == 3, "");

  constexpr bool incBool() {
    bool b = false;
    return ++b; // expected-error {{ISO C++17 does not allow incrementing expression of type bool}} \
                // ref-error {{ISO C++17 does not allow incrementing expression of type bool}}
  }
  static_assert(incBool(), "");

  /// FIXME: The diagnostics for pre-inc/dec of pointers doesn't match the
  /// current interpreter. But they are stil OK.
  template<typename T, bool Inc, bool Pre>
  constexpr int uninit() {
    T a;
    if constexpr (Inc) {
      if (Pre)
        ++a; // ref-note 3{{increment of uninitialized}} \
             // expected-note 2{{increment of uninitialized}} \
             // expected-note {{read of uninitialized}}
      else
        a++; // ref-note 2{{increment of uninitialized}} \
             // expected-note 2{{increment of uninitialized}}
    } else {
      if (Pre)
        --a; // ref-note 3{{decrement of uninitialized}} \
             // expected-note 2{{decrement of uninitialized}} \
             // expected-note {{read of uninitialized}}
      else
        a--; // ref-note 2{{decrement of uninitialized}} \
             // expected-note 2{{decrement of uninitialized}}
    }
    return 1;
  }
  static_assert(uninit<int, true, true>(), ""); // ref-error {{not an integral constant expression}} \
                                                // ref-note {{in call to 'uninit<int, true, true>()'}} \
                                                // expected-error {{not an integral constant expression}} \
                                                // expected-note {{in call to 'uninit()'}}
  static_assert(uninit<int, false, true>(), ""); // ref-error {{not an integral constant expression}} \
                                                 // ref-note {{in call to 'uninit<int, false, true>()'}} \
                                                 // expected-error {{not an integral constant expression}} \
                                                 // expected-note {{in call to 'uninit()'}}

  static_assert(uninit<float, true, true>(), ""); // ref-error {{not an integral constant expression}} \
                                                  // ref-note {{in call to 'uninit<float, true, true>()'}} \
                                                  // expected-error {{not an integral constant expression}} \
                                                  // expected-note {{in call to 'uninit()'}}
  static_assert(uninit<float, false, true>(), ""); // ref-error {{not an integral constant expression}} \
                                                   // ref-note {{in call to 'uninit<float, false, true>()'}} \
                                                   // expected-error {{not an integral constant expression}} \
                                                   // expected-note {{in call to 'uninit()'}}
  static_assert(uninit<float, true, false>(), ""); // ref-error {{not an integral constant expression}} \
                                                   // ref-note {{in call to 'uninit<float, true, false>()'}} \
                                                   // expected-error {{not an integral constant expression}} \
                                                   // expected-note {{in call to 'uninit()'}}
  static_assert(uninit<float, false, false>(), ""); // ref-error {{not an integral constant expression}} \
                                                    // ref-note {{in call to 'uninit<float, false, false>()'}} \
                                                    // expected-error {{not an integral constant expression}} \
                                                    // expected-note {{in call to 'uninit()'}}

  static_assert(uninit<int*, true, true>(), ""); // ref-error {{not an integral constant expression}} \
                                                 // ref-note {{in call to 'uninit<int *, true, true>()'}} \
                                                 // expected-error {{not an integral constant expression}} \
                                                 // expected-note {{in call to 'uninit()'}}
  static_assert(uninit<int*, false, true>(), ""); // ref-error {{not an integral constant expression}} \
                                                  // ref-note {{in call to 'uninit<int *, false, true>()'}} \
                                                  // expected-error {{not an integral constant expression}} \
                                                  // expected-note {{in call to 'uninit()'}}
  static_assert(uninit<int*, true, false>(), ""); // ref-error {{not an integral constant expression}} \
                                                  // ref-note {{in call to 'uninit<int *, true, false>()'}} \
                                                  // expected-error {{not an integral constant expression}} \
                                                  // expected-note {{in call to 'uninit()'}}
  static_assert(uninit<int*, false, false>(), ""); // ref-error {{not an integral constant expression}} \
                                                   // ref-note {{in call to 'uninit<int *, false, false>()'}} \
                                                   // expected-error {{not an integral constant expression}} \
                                                   // expected-note {{in call to 'uninit()'}}

  constexpr int OverFlow() { // ref-error {{never produces a constant expression}} \
                             // expected-error {{never produces a constant expression}}
    int a = INT_MAX;
    ++a; // ref-note 2{{is outside the range}} \
         // expected-note 2{{is outside the range}}
    return -1;
  }
  static_assert(OverFlow() == -1, "");  // expected-error {{not an integral constant expression}} \
                                        // expected-note {{in call to 'OverFlow()'}} \
                                        // ref-error {{not an integral constant expression}} \
                                        // ref-note {{in call to 'OverFlow()'}}


  constexpr int UnderFlow() { // ref-error {{never produces a constant expression}} \
                              // expected-error {{never produces a constant expression}}
    int a = INT_MIN;
    --a; // ref-note 2{{is outside the range}} \
         // expected-note 2{{is outside the range}}
    return -1;
  }
  static_assert(UnderFlow() == -1, "");  // expected-error {{not an integral constant expression}} \
                                         // expected-note {{in call to 'UnderFlow()'}} \
                                         // ref-error {{not an integral constant expression}} \
                                         // ref-note {{in call to 'UnderFlow()'}}

  constexpr int getTwo() {
    int i = 1;
    return (i += 1);
  }
  static_assert(getTwo() == 2, "");

  constexpr int sub(int a) {
    return (a -= 2);
  }
  static_assert(sub(7) == 5, "");

  constexpr int add(int a, int b) {
    a += b; // expected-note {{is outside the range of representable values}} \
            // ref-note {{is outside the range of representable values}} 
    return a;
  }
  static_assert(add(1, 2) == 3, "");
  static_assert(add(INT_MAX, 1) == 0, ""); // expected-error {{not an integral constant expression}} \
                                           // expected-note {{in call to 'add}} \
                                           // ref-error {{not an integral constant expression}} \
                                           // ref-note {{in call to 'add}}

  constexpr int sub(int a, int b) {
    a -= b; // expected-note {{is outside the range of representable values}} \
            // ref-note {{is outside the range of representable values}} 
    return a;
  }
  static_assert(sub(10, 20) == -10, "");
  static_assert(sub(INT_MIN, 1) == 0, ""); // expected-error {{not an integral constant expression}} \
                                           // expected-note {{in call to 'sub}} \
                                           // ref-error {{not an integral constant expression}} \
                                           // ref-note {{in call to 'sub}}

  constexpr int subAll(int a) {
    return (a -= a);
  }
  static_assert(subAll(213) == 0, "");

  constexpr bool BoolOr(bool b1, bool b2) {
    bool a;
    a = b1;
    a |= b2;
    return a;
  }
  static_assert(BoolOr(true, true), "");
  static_assert(BoolOr(true, false), "");
  static_assert(BoolOr(false, true), "");
  static_assert(!BoolOr(false, false), "");

  constexpr int IntOr(unsigned a, unsigned b) {
    unsigned r;
    r = a;
    r |= b;
    return r;
  }
  static_assert(IntOr(10, 1) == 11, "");
  static_assert(IntOr(1337, -1) == -1, "");
  static_assert(IntOr(0, 12) == 12, "");

  constexpr bool BoolAnd(bool b1, bool b2) {
    bool a;
    a = b1;
    a &= b2;
    return a;
  }
  static_assert(BoolAnd(true, true), "");
  static_assert(!BoolAnd(true, false), "");
  static_assert(!BoolAnd(false, true), "");
  static_assert(!BoolAnd(false, false), "");

  constexpr int IntAnd(unsigned a, unsigned b) {
    unsigned r;
    r = a;
    r &= b;
    return r;
  }
  static_assert(IntAnd(10, 1) == 0, "");
  static_assert(IntAnd(1337, -1) == 1337, "");
  static_assert(IntAnd(0, 12) == 0, "");

  constexpr bool BoolXor(bool b1, bool b2) {
    bool a;
    a = b1;
    a ^= b2;
    return a;
  }
  static_assert(!BoolXor(true, true), "");
  static_assert(BoolXor(true, false), "");
  static_assert(BoolXor(false, true), "");
  static_assert(!BoolXor(false, false), "");

  constexpr int IntXor(unsigned a, unsigned b) {
    unsigned r;
    r = a;
    r ^= b;
    return r;
  }
  static_assert(IntXor(10, 1) == 11, "");
  static_assert(IntXor(10, 10) == 0, "");
  static_assert(IntXor(12, true) == 13, "");

  constexpr bool BoolRem(bool b1, bool b2) {
    bool a;
    a = b1;
    a %= b2;
    return a;
  }
  static_assert(!BoolRem(true, true), "");
  static_assert(!BoolRem(false, true), "");

  constexpr int IntRem(int a, int b) {
    int r;
    r = a;
    r %= b; // expected-note {{division by zero}} \
            // ref-note {{division by zero}} \
            // expected-note {{outside the range of representable values}} \
            // ref-note {{outside the range of representable values}}
    return r;
  }
  static_assert(IntRem(2, 2) == 0, "");
  static_assert(IntRem(2, 1) == 0, "");
  static_assert(IntRem(9, 7) == 2, "");
  static_assert(IntRem(5, 0) == 0, ""); // expected-error {{not an integral constant expression}} \
                                        // expected-note {{in call to 'IntRem(5, 0)'}} \
                                        // ref-error {{not an integral constant expression}} \
                                        // ref-note {{in call to 'IntRem(5, 0)'}}

  static_assert(IntRem(INT_MIN, -1) == 0, ""); // expected-error {{not an integral constant expression}} \
                                               // expected-note {{in call to 'IntRem}} \
                                               // ref-error {{not an integral constant expression}} \
                                               // ref-note {{in call to 'IntRem}}



  constexpr bool BoolDiv(bool b1, bool b2) {
    bool a;
    a = b1;
    a /= b2;
    return a;
  }
  static_assert(BoolDiv(true, true), "");
  static_assert(!BoolDiv(false, true), "");

  constexpr int IntDiv(int a, int b) {
    int r;
    r = a;
    r /= b; // expected-note {{division by zero}} \
            // ref-note {{division by zero}} \
            // expected-note {{outside the range of representable values}} \
            // ref-note {{outside the range of representable values}}
    return r;
  }
  static_assert(IntDiv(2, 2) == 1, "");
  static_assert(IntDiv(12, 20) == 0, "");
  static_assert(IntDiv(2, 1) == 2, "");
  static_assert(IntDiv(9, 7) == 1, "");
  static_assert(IntDiv(5, 0) == 0, ""); // expected-error {{not an integral constant expression}} \
                                        // expected-note {{in call to 'IntDiv(5, 0)'}} \
                                        // ref-error {{not an integral constant expression}} \
                                        // ref-note {{in call to 'IntDiv(5, 0)'}}

  static_assert(IntDiv(INT_MIN, -1) == 0, ""); // expected-error {{not an integral constant expression}} \
                                               // expected-note {{in call to 'IntDiv}} \
                                               // ref-error {{not an integral constant expression}} \
                                               // ref-note {{in call to 'IntDiv}}

  constexpr bool BoolMul(bool b1, bool b2) {
    bool a;
    a = b1;
    a *= b2;
    return a;
  }
  static_assert(BoolMul(true, true), "");
  static_assert(!BoolMul(true, false), "");
  static_assert(!BoolMul(false, true), "");
  static_assert(!BoolMul(false, false), "");

  constexpr int IntMul(int a, int b) {
    int r;
    r = a;
    r *= b; // expected-note {{is outside the range of representable values of type 'int'}} \
            // ref-note {{is outside the range of representable values of type 'int'}}
    return r;
  }
  static_assert(IntMul(2, 2) == 4, "");
  static_assert(IntMul(12, 20) == 240, "");
  static_assert(IntMul(2, 1) == 2, "");
  static_assert(IntMul(9, 7) == 63, "");
  static_assert(IntMul(INT_MAX, 2) == 0, ""); // expected-error {{not an integral constant expression}} \
                                              // expected-note {{in call to 'IntMul}} \
                                              // ref-error {{not an integral constant expression}} \
                                              // ref-note {{in call to 'IntMul}}
  constexpr int arr[] = {1,2,3};
  constexpr int ptrInc1() {
    const int *p = arr;
    p += 2;
    return *p;
  }
  static_assert(ptrInc1() == 3, "");

  constexpr int ptrInc2() {
    const int *p = arr;
    return *(p += 1);
  }
  static_assert(ptrInc2() == 2, "");

  constexpr int ptrInc3() { // expected-error {{never produces a constant expression}} \
                            // ref-error {{never produces a constant expression}}
    const int *p = arr;
    p += 12; // expected-note {{cannot refer to element 12 of array of 3 elements}} \
             // ref-note {{cannot refer to element 12 of array of 3 elements}}
    return *p;
  }

  constexpr int ptrIncDec1() {
    const int *p = arr;
    p += 2;
    p -= 1;
    return *p;
  }
  static_assert(ptrIncDec1() == 2, "");

  constexpr int ptrDec1() { // expected-error {{never produces a constant expression}} \
                        // ref-error {{never produces a constant expression}}
    const int *p = arr;
    p -= 1;  // expected-note {{cannot refer to element -1 of array of 3 elements}} \
             // ref-note {{cannot refer to element -1 of array of 3 elements}}
    return *p;
  }

  /// This used to leave a 0 on the stack instead of the previous
  /// value of a.
  constexpr int bug1Inc() {
    int a = 3;
    int b = a++;
    return b;
  }
  static_assert(bug1Inc() == 3);

  constexpr int bug1Dec() {
    int a = 3;
    int b = a--;
    return b;
  }
  static_assert(bug1Dec() == 3);

  constexpr int f() {
    int a[] = {1,2};
    int i = 0;

    // RHS should be evaluated before LHS, so this should
    // write to a[1];
    a[i++] += ++i;

    return a[1];
  }
  static_assert(f() == 3, "");
};
#endif

namespace CompoundLiterals {
  constexpr int get5() {
    return (int[]){1,2,3,4,5}[4];
  }
  static_assert(get5() == 5, "");

  constexpr int get6(int f = (int[]){1,2,6}[2]) { // ref-note {{subexpression not valid in a constant expression}} \
                                                  // ref-note {{declared here}}
    return f;
  }
  static_assert(get6(6) == 6, "");
  // FIXME: Who's right here?
  static_assert(get6() == 6, ""); // ref-error {{not an integral constant expression}}

  constexpr int x = (int){3};
  static_assert(x == 3, "");
#if __cplusplus >= 201402L
  constexpr int getX() {
    int x = (int){3};
    x = (int){5};
    return x;
  }
  static_assert(getX() == 5, "");
#endif

#if __cplusplus >= 202002L
  constexpr int get3() {
    int m;
    m = (int){3};
    return m;
  }
  static_assert(get3() == 3, "");
#endif
};

namespace TypeTraits {
  static_assert(__is_trivial(int), "");
  static_assert(__is_trivial(float), "");
  static_assert(__is_trivial(E), "");
  struct S{};
  static_assert(__is_trivial(S), "");
  struct S2 {
    S2() {}
  };
  static_assert(!__is_trivial(S2), "");

  template <typename T>
  struct S3 {
    constexpr bool foo() const { return __is_trivial(T); }
  };
  struct T {
    ~T() {}
  };
  struct U {};
  static_assert(S3<U>{}.foo(), "");
  static_assert(!S3<T>{}.foo(), "");
}

#if __cplusplus >= 201402L
constexpr int ignoredDecls() {
  static_assert(true, "");
  struct F { int a; };
  enum E { b };
  using A = int;
  typedef int Z;

  return F{12}.a;
}
static_assert(ignoredDecls() == 12, "");

namespace DiscardExprs {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"

  struct A{ int a; };
  constexpr int ignoredExprs() {
    (void)(1 / 2);
    A a{12};
    a;
    (void)a;
    (a);

    /// Ignored MaterializeTemporaryExpr.
    struct B{ const int &a; };
    (void)B{12};

    (void)5, (void)6;

    1 ? 0 : 1;
    __is_trivial(int);

    (int){1};
    (int[]){1,2,3};
    int arr[] = {1,2,3};
    arr[0];
    "a";
    'b';
    sizeof(int);
    alignof(int);

    (short)5;
    (bool)1;
    __null;
    __builtin_offsetof(A, a);
    1,2;

    return 0;
  }
  static_assert(ignoredExprs() == 0, "");

  constexpr int oh_my(int x) {
    (int){ x++ };
    return x;
  }
  static_assert(oh_my(0) == 1, "");

  constexpr int oh_my2(int x) {
    int y{x++};
    return x;
  }

  static_assert(oh_my2(0) == 1, "");


  /// Ignored comma expressions still have their
  /// expressions evaluated.
  constexpr int Comma(int start) {
      int i = start;

      (void)i++;
      (void)i++,(void)i++;
      return i;
  }
  constexpr int Value = Comma(5);
  static_assert(Value == 8, "");

  /// Ignored MemberExprs need to still evaluate the Base
  /// expr.
  constexpr A callme(int &i) {
    ++i;
    return A{};
  }
  constexpr int ignoredMemberExpr() {
    int i = 0;
    callme(i).a;
    return i;
  }
  static_assert(ignoredMemberExpr() == 1, "");

  template <int I>
  constexpr int foo() {
    I;
    return I;
  }
  static_assert(foo<3>() == 3, "");

#pragma clang diagnostic pop
}
#endif

namespace PredefinedExprs {
#if __cplusplus >= 201402L
  template<typename CharT>
  constexpr bool strings_match(const CharT *str1, const CharT *str2) {
    while (*str1 && *str2) {
      if (*str1++ != *str2++)
        return false;
    };

    return *str1 == *str2;
  }

  void foo() {
    static_assert(strings_match(__FUNCSIG__, "void __cdecl PredefinedExprs::foo(void)"), "");
    static_assert(strings_match(L__FUNCSIG__, L"void __cdecl PredefinedExprs::foo(void)"), "");
    static_assert(strings_match(L__FUNCTION__, L"foo"), "");
    static_assert(strings_match(__FUNCTION__, "foo"), "");
    static_assert(strings_match(__func__, "foo"), "");
    static_assert(strings_match(__PRETTY_FUNCTION__, "void PredefinedExprs::foo()"), "");
  }

  constexpr char heh(unsigned index) {
    __FUNCTION__;               // ref-warning {{result unused}} \
                                // expected-warning {{result unused}}
    __extension__ __FUNCTION__; // ref-warning {{result unused}} \
                                // expected-warning {{result unused}}
    return __FUNCTION__[index];
  }
  static_assert(heh(0) == 'h', "");
  static_assert(heh(1) == 'e', "");
  static_assert(heh(2) == 'h', "");
#endif
}

namespace NE {
  constexpr int foo() noexcept {
    return 1;
  }
  static_assert(noexcept(foo()), "");
  constexpr int foo2() {
    return 1;
  }
  static_assert(!noexcept(foo2()), "");

#if __cplusplus > 201402L
  constexpr int a() {
    int b = 0;
    (void)noexcept(++b); // expected-warning {{expression with side effects has no effect in an unevaluated context}} \
                         // ref-warning {{expression with side effects has no effect in an unevaluated context}}

    return b;
  }
  static_assert(a() == 0, "");
#endif
}

namespace PointerCasts {
  constexpr int M = 10;
  constexpr const int *P = &M;
  constexpr intptr_t A = (intptr_t)P; // ref-error {{must be initialized by a constant expression}} \
                                      // ref-note {{cast that performs the conversions of a reinterpret_cast}} \
                                      // expected-error {{must be initialized by a constant expression}} \
                                      // expected-note {{cast that performs the conversions of a reinterpret_cast}}

  int array[(intptr_t)(char*)0]; // ref-warning {{variable length array folded to constant array}} \
                                 // expected-warning {{variable length array folded to constant array}}
}

namespace InvalidDeclRefs {
  bool b00; // ref-note {{declared here}} \
            // expected-note {{declared here}}
  static_assert(b00, ""); // ref-error {{not an integral constant expression}} \
                          // ref-note {{read of non-const variable}} \
                          // expected-error {{not an integral constant expression}} \
                          // expected-note {{read of non-const variable}}

  float b01; // ref-note {{declared here}} \
             // expected-note {{declared here}}
  static_assert(b01, ""); // ref-error {{not an integral constant expression}} \
                          // ref-note {{read of non-constexpr variable}} \
                          // expected-error {{not an integral constant expression}} \
                          // expected-note {{read of non-constexpr variable}}

  extern const int b02; // ref-note {{declared here}} \
                        // expected-note {{declared here}}
  static_assert(b02, ""); // ref-error {{not an integral constant expression}} \
                          // ref-note {{initializer of 'b02' is unknown}} \
                          // expected-error {{not an integral constant expression}} \
                          // expected-note {{initializer of 'b02' is unknown}}

  /// FIXME: This should also be diagnosed in the new interpreter.
  int b03 = 3; // ref-note {{declared here}}
  static_assert(b03, ""); // ref-error {{not an integral constant expression}} \
                          // ref-note {{read of non-const variable}}
}
