// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -Wno-vla -fms-extensions -std=c++11 -verify=expected,both %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -Wno-vla -fms-extensions -std=c++20 -verify=expected,both %s
// RUN: %clang_cc1 -std=c++11 -fms-extensions -Wno-vla -verify=ref,both %s
// RUN: %clang_cc1 -std=c++20 -fms-extensions -Wno-vla -verify=ref,both %s

#define INT_MIN (~__INT_MAX__)
#define INT_MAX __INT_MAX__

typedef __INTPTR_TYPE__ intptr_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;


static_assert(true, "");
static_assert(false, ""); // both-error{{failed}}
static_assert(nullptr == nullptr, "");
static_assert(__null == __null, "");
static_assert(1 == 1, "");
static_assert(1 == 3, ""); // both-error{{failed}}

constexpr void* v = nullptr;
static_assert(__null == v, "");

constexpr int number = 10;
static_assert(number == 10, "");
static_assert(number != 10, ""); // both-error{{failed}} \
                                 // both-note{{evaluates to}}

static_assert(__objc_yes, "");
static_assert(!__objc_no, "");

constexpr bool b = number;
static_assert(b, "");
constexpr int one = true;
static_assert(one == 1, "");

constexpr bool b2 = bool();
static_assert(!b2, "");

constexpr int Failed1 = 1 / 0; // both-error {{must be initialized by a constant expression}} \
                               // both-note {{division by zero}} \
                               // both-note {{declared here}}
constexpr int Failed2 = Failed1 + 1; // both-error {{must be initialized by a constant expression}} \
                                     // both-note {{declared here}} \
                                     // both-note {{initializer of 'Failed1' is not a constant expression}}
static_assert(Failed2 == 0, ""); // both-error {{not an integral constant expression}} \
                                 // both-note {{initializer of 'Failed2' is not a constant expression}}

const int x = *(volatile int*)0x1234;
static_assert((void{}, true), "");

namespace ScalarTypes {
  constexpr int ScalarInitInt = int();
  static_assert(ScalarInitInt == 0, "");
  constexpr float ScalarInitFloat = float();
  static_assert(ScalarInitFloat == 0.0f, "");

  static_assert(decltype(nullptr)() == nullptr, "");

  template<typename T>
  constexpr T getScalar() { return T(); }

  static_assert(getScalar<const int>() == 0, "");
  static_assert(getScalar<const double>() == 0.0, "");

  static_assert(getScalar<void*>() == nullptr, "");
  static_assert(getScalar<void(*)(void)>() == nullptr, "");

  enum E {
    First = 0,
  };
  static_assert(getScalar<E>() == First, "");

  struct S {
    int v;
  };
  constexpr int S::* MemberPtr = &S::v;
  static_assert(getScalar<decltype(MemberPtr)>() == nullptr, "");

#if __cplusplus >= 201402L
  constexpr void Void(int n) {
    void(n + 1);
    void();
  }
  constexpr int void_test = (Void(0), 1);
  static_assert(void_test == 1, "");
#endif
}

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

constexpr int UninitI; // both-error {{must be initialized by a constant expression}}
constexpr int *UninitPtr; // both-error {{must be initialized by a constant expression}}

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
static_assert(-false, ""); //both-error{{failed}}

static_assert(~0 == -1, "");
static_assert(~1 == -2, "");
static_assert(~-1 == 0, "");
static_assert(~255 == -256, "");
static_assert(~INT_MIN == INT_MAX, "");
static_assert(~INT_MAX == INT_MIN, "");

static_assert(-(1 << 31), ""); // both-error {{not an integral constant expression}} \
                               // both-note {{outside the range of representable values}}

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
  constexpr bool v2 = null < pv; // both-error {{must be initialized by a constant expression}} \
                                 // both-note {{comparison between pointers to unrelated objects 'nullptr' and '&s.a' has unspecified value}}

  constexpr bool v3 = null == pv; // ok
  constexpr bool v4 = qv == pv; // ok

  constexpr bool v5 = qv >= pv;
  constexpr bool v8 = qv > (void*)&s.a;
  constexpr bool v6 = qv > null; // both-error {{must be initialized by a constant expression}} \
                                 // both-note {{comparison between pointers to unrelated objects '&s.b' and 'nullptr' has unspecified value}}

  constexpr bool v7 = qv <= (void*)&s.b; // ok

  constexpr ptrdiff_t m = &m - &m;
  static_assert(m == 0, "");

  constexpr ptrdiff_t m2 = (&m2 + 1) - (&m2 + 1);
  static_assert(m2 == 0, "");

  constexpr long m3 = (&m3 + 1) - (&m3);
  static_assert(m3 == 1, "");

  constexpr long m4 = &m4 + 2 - &m4; // both-error {{must be initialized by a constant expression}} \
                                     // both-note {{cannot refer to element 2 of non-array object}}
}

namespace SizeOf {
  static_assert(alignof(char&) == 1, "");

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

  constexpr int F = sizeof(void); // both-error{{incomplete type 'void'}}

  constexpr int F2 = sizeof(gimme); // both-error{{to a function type}}


  struct S {
    void func();
  };
  constexpr void (S::*Func)() = &S::func;
  static_assert(sizeof(Func) == sizeof(&S::func), "");


  void func() {
    int n = 12;
    constexpr int oofda = sizeof(int[n++]); // both-error {{must be initialized by a constant expression}}
  }

#if __cplusplus >= 201402L
  constexpr int IgnoredRejected() { // both-error {{never produces a constant expression}}
    int n = 0;
    sizeof(int[n++]); // both-warning {{expression result unused}} \
                      // both-note 2{{subexpression not valid in a constant expression}}
    return n;
  }
  static_assert(IgnoredRejected() == 0, ""); // both-error {{not an integral constant expression}} \
                                             // both-note {{in call to 'IgnoredRejected()'}}
#endif


#if __cplusplus >= 202002L
  /// FIXME: The following code should be accepted.
  consteval int foo(int n) { // both-error {{consteval function never produces a constant expression}}
    return sizeof(int[n]); // both-note 3{{not valid in a constant expression}}
  }
  constinit int var = foo(5); // both-error {{not a constant expression}} \
                              // both-note 2{{in call to}} \
                              // both-error {{does not have a constant initializer}} \
                              // both-note {{required by 'constinit' specifier}}

#endif
};

namespace rem {
  static_assert(2 % 2 == 0, "");
  static_assert(2 % 1 == 0, "");
  static_assert(-3 % 4 == -3, "");
  static_assert(4 % -2 == 0, "");
  static_assert(-3 % -4 == -3, "");

  constexpr int zero() { return 0; }
  static_assert(10 % zero() == 20, ""); // both-error {{not an integral constant expression}} \
                                        // both-note {{division by zero}}

  static_assert(true % true == 0, "");
  static_assert(false % true == 0, "");
  static_assert(true % false == 10, ""); // both-error {{not an integral constant expression}} \
                                         // both-note {{division by zero}}
  constexpr int x = INT_MIN % - 1; // both-error {{must be initialized by a constant expression}} \
                                   // both-note {{value 2147483648 is outside the range}}
};

namespace div {
  constexpr int zero() { return 0; }
  static_assert(12 / 3 == 4, "");
  static_assert(12 / zero() == 12, ""); // both-error {{not an integral constant expression}} \
                                        // both-note {{division by zero}}
  static_assert(12 / -3 == -4, "");
  static_assert(-12 / 3 == -4, "");


  constexpr int LHS = 12;
  constexpr long unsigned RHS = 3;
  static_assert(LHS / RHS == 4, "");

  constexpr int x = INT_MIN / - 1; // both-error {{must be initialized by a constant expression}} \
                                   // both-note {{value 2147483648 is outside the range}}
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
  !bo; // both-warning {{expression result unused}}
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
  __WCHAR_TYPE__ wm = L'abc'; // both-error{{wide character literals may not contain multiple characters}}
  __WCHAR_TYPE__ wu = u'abc'; // both-error{{Unicode character literals may not contain multiple characters}}
  __WCHAR_TYPE__ wU = U'abc'; // both-error{{Unicode character literals may not contain multiple characters}}
#if __cplusplus > 201103L
  __WCHAR_TYPE__ wu8 = u8'abc'; // both-error{{Unicode character literals may not contain multiple characters}}
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
  static_assert(foo2[8] == '\0', ""); // both-error {{not an integral constant expression}} \
                                      // both-note {{read of dereferenced one-past-the-end pointer}}

  constexpr char foo3[4] = "abc";
  static_assert(foo3[3] == '\0', "");
  static_assert(foo3[4] == '\0', ""); // both-error {{not an integral constant expression}} \
                                      // both-note {{read of dereferenced one-past-the-end pointer}}

  constexpr char foo4[2] = "abcd"; // both-error {{initializer-string for char array is too long}}
  static_assert(foo4[0] == 'a', "");
  static_assert(foo4[1] == 'b', "");
  static_assert(foo4[2] == '\0', ""); // both-error {{not an integral constant expression}} \
                                      // both-note {{read of dereferenced one-past-the-end pointer}}

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
    return ++a + ++a; // both-warning {{multiple unsequenced modifications to 'a'}}
  }
  static_assert(three() == 3, "");

  constexpr bool incBool() {
    bool b = false;
    return ++b; // both-error {{ISO C++17 does not allow incrementing expression of type bool}}
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
  static_assert(uninit<int, true, true>(), ""); // both-error {{not an integral constant expression}} \
                                                // both-note {{in call to 'uninit<int, true, true>()'}}
  static_assert(uninit<int, false, true>(), ""); // both-error {{not an integral constant expression}} \
                                                 // both-note {{in call to 'uninit<int, false, true>()'}}

  static_assert(uninit<float, true, true>(), ""); // both-error {{not an integral constant expression}} \
                                                  // both-note {{in call to 'uninit<float, true, true>()'}}
  static_assert(uninit<float, false, true>(), ""); // both-error {{not an integral constant expression}} \
                                                   // both-note {{in call to 'uninit<float, false, true>()'}}
  static_assert(uninit<float, true, false>(), ""); // both-error {{not an integral constant expression}} \
                                                   // both-note {{in call to 'uninit<float, true, false>()'}}
  static_assert(uninit<float, false, false>(), ""); // both-error {{not an integral constant expression}} \
                                                    // both-note {{in call to 'uninit<float, false, false>()'}}

  static_assert(uninit<int*, true, true>(), ""); // both-error {{not an integral constant expression}} \
                                                 // both-note {{in call to 'uninit<int *, true, true>()'}}
  static_assert(uninit<int*, false, true>(), ""); // both-error {{not an integral constant expression}} \
                                                  // both-note {{in call to 'uninit<int *, false, true>()'}}
  static_assert(uninit<int*, true, false>(), ""); // both-error {{not an integral constant expression}} \
                                                  // both-note {{in call to 'uninit<int *, true, false>()'}}
  static_assert(uninit<int*, false, false>(), ""); // both-error {{not an integral constant expression}} \
                                                   // both-note {{in call to 'uninit<int *, false, false>()'}}

  constexpr int OverFlow() { // both-error {{never produces a constant expression}}
    int a = INT_MAX;
    ++a; // both-note 2{{is outside the range}}
    return -1;
  }
  static_assert(OverFlow() == -1, "");  // both-error {{not an integral constant expression}} \
                                        // both-note {{in call to 'OverFlow()'}}

  constexpr int UnderFlow() { // both-error {{never produces a constant expression}}
    int a = INT_MIN;
    --a; // both-note 2{{is outside the range}}
    return -1;
  }
  static_assert(UnderFlow() == -1, "");  // both-error {{not an integral constant expression}} \
                                         // both-note {{in call to 'UnderFlow()'}}

  /// This UnaryOperator can't overflow, so we shouldn't diagnose any overflow.
  constexpr int CanOverflow() {
    char c = 127;
    char p;
    ++c;
    c++;
    p = ++c;
    p = c++;

    c = -128;
    --c;
    c--;
    p = --c;
    p = ++c;

    return 0;
  }
  static_assert(CanOverflow() == 0, "");

  constexpr char OverflownChar() {
    char c = 127;
    c++;
    return c;
  }
  static_assert(OverflownChar() == -128, "");

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
    a += b; // both-note {{is outside the range of representable values}}
    return a;
  }
  static_assert(add(1, 2) == 3, "");
  static_assert(add(INT_MAX, 1) == 0, ""); // both-error {{not an integral constant expression}} \
                                           // both-note {{in call to 'add}}

  constexpr int sub(int a, int b) {
    a -= b; // both-note {{is outside the range of representable values}}
    return a;
  }
  static_assert(sub(10, 20) == -10, "");
  static_assert(sub(INT_MIN, 1) == 0, ""); // both-error {{not an integral constant expression}} \
                                           // both-note {{in call to 'sub}}

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
    r %= b; // both-note {{division by zero}} \
            // both-note {{outside the range of representable values}}
    return r;
  }
  static_assert(IntRem(2, 2) == 0, "");
  static_assert(IntRem(2, 1) == 0, "");
  static_assert(IntRem(9, 7) == 2, "");
  static_assert(IntRem(5, 0) == 0, ""); // both-error {{not an integral constant expression}} \
                                        // both-note {{in call to 'IntRem(5, 0)'}}

  static_assert(IntRem(INT_MIN, -1) == 0, ""); // both-error {{not an integral constant expression}} \
                                               // both-note {{in call to 'IntRem}}

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
    r /= b; // both-note {{division by zero}} \
            // both-note {{outside the range of representable values}}
    return r;
  }
  static_assert(IntDiv(2, 2) == 1, "");
  static_assert(IntDiv(12, 20) == 0, "");
  static_assert(IntDiv(2, 1) == 2, "");
  static_assert(IntDiv(9, 7) == 1, "");
  static_assert(IntDiv(5, 0) == 0, ""); // both-error {{not an integral constant expression}} \
                                        // both-note {{in call to 'IntDiv(5, 0)'}}

  static_assert(IntDiv(INT_MIN, -1) == 0, ""); // both-error {{not an integral constant expression}} \
                                               // both-note {{in call to 'IntDiv}}

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
    r *= b; // both-note {{is outside the range of representable values of type 'int'}}
    return r;
  }
  static_assert(IntMul(2, 2) == 4, "");
  static_assert(IntMul(12, 20) == 240, "");
  static_assert(IntMul(2, 1) == 2, "");
  static_assert(IntMul(9, 7) == 63, "");
  static_assert(IntMul(INT_MAX, 2) == 0, ""); // both-error {{not an integral constant expression}} \
                                              // both-note {{in call to 'IntMul}}
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

  constexpr int ptrInc3() { // both-error {{never produces a constant expression}}
    const int *p = arr;
    p += 12; // both-note {{cannot refer to element 12 of array of 3 elements}}
    return *p;
  }

  constexpr int ptrIncDec1() {
    const int *p = arr;
    p += 2;
    p -= 1;
    return *p;
  }
  static_assert(ptrIncDec1() == 2, "");

  constexpr int ptrDec1() { // both-error {{never produces a constant expression}}
    const int *p = arr;
    p -= 1;  // both-note {{cannot refer to element -1 of array of 3 elements}}
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

  int nonconst(int a) { // both-note 4{{declared here}}
    static_assert(a++, ""); // both-error {{not an integral constant expression}} \
                            // both-note {{function parameter 'a' with unknown value cannot be used in a constant expression}}
    static_assert(a--, ""); // both-error {{not an integral constant expression}} \
                            // both-note {{function parameter 'a' with unknown value cannot be used in a constant expression}}
    static_assert(++a, ""); // both-error {{not an integral constant expression}} \
                            // both-note {{function parameter 'a' with unknown value cannot be used in a constant expression}}
    static_assert(--a, ""); // both-error {{not an integral constant expression}} \
                            // both-note {{function parameter 'a' with unknown value cannot be used in a constant expression}}
  }

};
#endif

namespace CompoundLiterals {
  constexpr int get5() {
    return (int[]){1,2,3,4,5}[4];
  }
  static_assert(get5() == 5, "");

  constexpr int get6(int f = (int[]){1,2,6}[2]) {
    return f;
  }
  static_assert(get6(6) == 6, "");
  static_assert(get6() == 6, "");

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

  constexpr int *f(int *a=(int[]){1,2,3}) { return a; } // both-note {{temporary created here}}
  constinit int *a1 = f(); // both-error {{variable does not have a constant initializer}} \
                              both-note {{required by 'constinit' specifier here}} \
                              both-note {{pointer to subobject of temporary is not a constant expression}}
  static_assert(f()[0] == 1); // Ok
#endif

  constexpr int f2(int *x =(int[]){1,2,3}) {
    return x[0];
  }
  // Should evaluate to 1?
  constexpr int g = f2(); // #g_decl
  static_assert(g == 1, "");

  // This example should be rejected because the lifetime of the compound
  // literal assigned into x is that of the full expression, which is the
  // parenthesized assignment operator. So the return statement is using a
  // dangling pointer. FIXME: the note saying it's a read of a dereferenced
  // null pointer suggests we're doing something odd during constant expression
  // evaluation: I think it's still taking 'x' as being null from the call to
  // f3() rather than tracking the assignment happening in the VLA.
  constexpr int f3(int *x, int (*y)[*(x=(int[]){1,2,3})]) { // both-warning {{object backing the pointer 'x' will be destroyed at the end of the full-expression}}
    return x[0]; // both-note {{read of dereferenced null pointer is not allowed in a constant expression}}
  }
  constexpr int h = f3(0,0); // both-error {{constexpr variable 'h' must be initialized by a constant expression}} \
                                both-note {{in call to 'f3(nullptr, nullptr)'}}
}

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

  typedef int Int;
  typedef Int IntAr[10];
  typedef const IntAr ConstIntAr;
  typedef ConstIntAr ConstIntArAr[4];

  static_assert(__array_rank(IntAr) == 1, "");
  static_assert(__array_rank(ConstIntArAr) == 2, "");

  static_assert(__array_extent(IntAr, 0) == 10, "");
  static_assert(__array_extent(ConstIntArAr, 0) == 4, "");
  static_assert(__array_extent(ConstIntArAr, 1) == 10, "");
}

#if __cplusplus >= 201402L
namespace SomeNS {
  using MyInt = int;
}

constexpr int ignoredDecls() {
  static_assert(true, "");
  struct F { int a; };
  enum E { b };
  using A = int;
  typedef int Z;
  namespace NewNS = SomeNS;
  using NewNS::MyInt;

  return F{12}.a;
}
static_assert(ignoredDecls() == 12, "");

namespace DiscardExprs {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
  typedef struct _GUID {
    __UINT32_TYPE__ Data1;
    __UINT16_TYPE__ Data2;
    __UINT16_TYPE__ Data3;
    __UINT8_TYPE__ Data4[8];
  } GUID;
  class __declspec(uuid("000000A0-0000-0000-C000-000000000049")) GuidType;

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
    (int)1.0;
    (float)1;
    (double)1.0f;
    (signed)4u;
    __uuidof(GuidType);
    __uuidof(number); // both-error {{cannot call operator __uuidof on a type with no GUID}}

    requires{false;};
    constexpr int *p = nullptr;
    p - p;

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

  struct ATemp {
    consteval ATemp ret_a() const { return ATemp{}; }
  };

  void test() {
    int k = (ATemp().ret_a(), 0);
  }

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
    __FUNCTION__;               // both-warning {{result unused}}
    __extension__ __FUNCTION__; // both-warning {{result unused}}
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
    (void)noexcept(++b); // both-warning {{expression with side effects has no effect in an unevaluated context}}

    return b;
  }
  static_assert(a() == 0, "");
#endif
}

namespace PointerCasts {
  constexpr int M = 10;
  constexpr const int *P = &M;
  constexpr intptr_t A = (intptr_t)P; // both-error {{must be initialized by a constant expression}} \
                                      // both-note {{cast that performs the conversions of a reinterpret_cast}}

  int array[(intptr_t)(char*)0]; // both-warning {{variable length array folded to constant array}}
}

namespace InvalidDeclRefs {
  bool b00; // both-note {{declared here}}
  static_assert(b00, ""); // both-error {{not an integral constant expression}} \
                          // both-note {{read of non-const variable}}

  float b01; // both-note {{declared here}}
  static_assert(b01, ""); // both-error {{not an integral constant expression}} \
                          // both-note {{read of non-constexpr variable}}

  extern const int b02; // both-note {{declared here}}
  static_assert(b02, ""); // both-error {{not an integral constant expression}} \
                          // both-note {{initializer of 'b02' is unknown}}

  int b03 = 3; // both-note {{declared here}}
  static_assert(b03, ""); // both-error {{not an integral constant expression}} \
                          // both-note {{read of non-const variable}}

  extern int var;
  constexpr int *varp = &var; // Ok.
}

namespace NonConstReads {
  void *p = nullptr; // both-note {{declared here}}
  static_assert(!p, ""); // both-error {{not an integral constant expression}} \
                         // both-note {{read of non-constexpr variable 'p'}}

  int arr[!p]; // both-error {{variable length array}}

  int z; // both-note {{declared here}}
  static_assert(z == 0, ""); // both-error {{not an integral constant expression}} \
                             // both-note {{read of non-const variable 'z'}}
}

/// This test passes a MaterializedTemporaryExpr to evaluateAsRValue.
/// That needs to return a null pointer after the lvalue-to-rvalue conversion.
/// We used to fail to do that.
namespace rdar8769025 {
  __attribute__((nonnull)) void f1(int * const &p);
  void test_f1() {
    f1(0); // both-warning{{null passed to a callee that requires a non-null argument}}
  }
}

namespace nullptrsub {
  void a() {
    char *f = (char *)0;
    f = (char *)((char *)0 - (char *)0);
  }
}

namespace incdecbool {
#if __cplusplus >= 201402L
  constexpr bool incb(bool c) {
    if (!c)
      ++c;
    else {++c; c++; }
#if __cplusplus >= 202002L
    // both-error@-3 {{ISO C++17 does not allow incrementing expression of type bool}}
    // both-error@-3 2{{ISO C++17 does not allow incrementing expression of type bool}}
#else
    // both-warning@-6 {{incrementing expression of type bool is deprecated and incompatible with C++17}}
#endif
    return c;
  }
  static_assert(incb(false), "");
  static_assert(incb(true), "");
  static_assert(incb(true) == 1, "");
#endif


#if __cplusplus == 201103L
  constexpr bool foo() { // both-error {{never produces a constant expression}}
    bool b = true; // both-warning {{variable declaration in a constexpr function is a C++14 extension}}
    b++; // both-warning {{incrementing expression of type bool is deprecated and incompatible with C++17}} \
         // both-warning {{use of this statement in a constexpr function is a C++14 extension}} \
         // both-note 2{{subexpression not valid in a constant expression}}

    return b;
  }
  static_assert(foo() == 1, ""); // both-error {{not an integral constant expression}} \
                                 // both-note {{in call to}}
#endif



}

#if __cplusplus >= 201402L
constexpr int externvar1() { // both-error {{never produces a constant expression}}
  extern char arr[]; // both-note {{declared here}}
   return arr[0]; // both-note {{read of non-constexpr variable 'arr'}}
}
namespace externarr {
  extern int arr[];
  constexpr int *externarrindex = &arr[0]; /// No diagnostic.
}


namespace StmtExprs {
  constexpr int foo() {
     ({
       int i;
       for (i = 0; i < 76; i++) {}
       i; // both-warning {{expression result unused}}
    });
    return 76;
  }
  static_assert(foo() == 76, "");

  namespace CrossFuncLabelDiff {
    constexpr long a(bool x) { return x ? 0 : (intptr_t)&&lbl + (0 && ({lbl: 0;})); }
  }
}
#endif

namespace Extern {
  constexpr extern char Oops = 1;
  static_assert(Oops == 1, "");

#if __cplusplus >= 201402L
  struct NonLiteral {
    NonLiteral() {}
  };
  NonLiteral nl;
  constexpr NonLiteral &ExternNonLiteralVarDecl() {
    extern NonLiteral nl;
    return nl;
  }
  static_assert(&ExternNonLiteralVarDecl() == &nl, "");
#endif

  struct A {
    int b;
  };

  extern constexpr A a{12};
  static_assert(a.b == 12, "");
}

#if __cplusplus >= 201402L
constexpr int StmtExprEval() {
  if (({
    while (0);
    true;
  })) {
    return 2;
  }
  return 1;
}
static_assert(StmtExprEval() == 2, "");

constexpr int ReturnInStmtExpr() { // both-error {{never produces a constant expression}}
  return ({
      return 1; // both-note 2{{this use of statement expressions is not supported in a constant expression}}
      2;
      });
}
static_assert(ReturnInStmtExpr() == 1, ""); // both-error {{not an integral constant expression}} \
                                            // both-note {{in call to}}

#endif

namespace ComparisonAgainstOnePastEnd {
  int a, b;
  static_assert(&a + 1 == &b, ""); // both-error {{not an integral constant expression}} \
                                   // both-note {{comparison against pointer '&a + 1' that points past the end of a complete object has unspecified value}}
  static_assert(&a == &b + 1, ""); // both-error {{not an integral constant expression}} \
                                   // both-note {{comparison against pointer '&b + 1' that points past the end of a complete object has unspecified value}}

  static_assert(&a + 1 == &b + 1, ""); // both-error {{static assertion failed}}
};

namespace NTTP {
  template <typename _Tp, unsigned _Nm>
    constexpr unsigned
    size(const _Tp (&)[_Nm]) noexcept
    { return _Nm; }

  template <char C>
  static int write_padding() {
    static const char Chars[] = {C};

    return size(Chars);
  }
}

#if __cplusplus >= 201402L
namespace UnaryOpError {
  constexpr int foo() {
    int f = 0;
    ++g; // both-error {{use of undeclared identifier 'g'}} \
            both-error {{cannot assign to variable 'g' with const-qualified type 'const int'}} \
            both-note@#g_decl {{'CompoundLiterals::g' declared here}} \
            both-note@#g_decl {{variable 'g' declared const here}}
    return f;
  }
}
#endif

namespace VolatileReads {
  const volatile int b = 1;
  static_assert(b, ""); // both-error {{not an integral constant expression}} \
                        // both-note {{read of volatile-qualified type 'const volatile int' is not allowed in a constant expression}}


  constexpr int a = 12;
  constexpr volatile int c = (volatile int&)a; // both-error {{must be initialized by a constant expression}} \
                                               // both-note {{read of volatile-qualified type 'volatile int'}}

  volatile constexpr int n1 = 0; // both-note {{here}}
  volatile const int n2 = 0; // both-note {{here}}
  constexpr int m1 = n1; // both-error {{constant expression}} \
                         // both-note {{read of volatile-qualified type 'const volatile int'}}
  constexpr int m2 = n2; // both-error {{constant expression}} \
                         // both-note {{read of volatile-qualified type 'const volatile int'}}
  constexpr int m1b = const_cast<const int&>(n1); // both-error {{constant expression}} \
                                                  // both-note {{read of volatile object 'n1'}}
  constexpr int m2b = const_cast<const int&>(n2); // both-error {{constant expression}} \
                                                  // both-note {{read of volatile object 'n2'}}

  struct S {
    constexpr S(int=0) : i(1) {}
    int i;
  };
  constexpr volatile S vs; // both-note {{here}}
  static_assert(const_cast<int&>(vs.i), ""); // both-error {{constant expression}} \
                                             // both-note {{read of volatile object 'vs'}}
}
#if __cplusplus >= 201703L
namespace {
  struct C {
    int x;
  };

  template <const C *p> void f() {
    const auto &[c] = *p;
    &c; // both-warning {{expression result unused}}
  }
}
#endif

void localConstexpr() {
  constexpr int a = 1/0; // both-error {{must be initialized by a constant expression}} \
                         // both-note {{division by zero}} \
                         // both-warning {{division by zero is undefined}} \
                         // both-note {{declared here}}
  static_assert(a == 0, ""); // both-error {{not an integral constant expression}} \
                             // both-note {{initializer of 'a' is not a constant expression}}
}

namespace Foo {
  namespace Bar {
    constexpr int FB = 10;
  }
}
constexpr int usingDirectiveDecl() {
  using namespace Foo::Bar;
  return FB;
}
static_assert(usingDirectiveDecl() == 10, "");
