// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -S -triple=x86_64 -mllvm -sort-timers=0 -o - -ftime-report  2>&1 | FileCheck %s
// RUN: %clang_cc1 %s -S -triple=x86_64 -mllvm -sort-timers=0 -o - -fdelayed-template-parsing -DDELAYED_TEMPLATE_PARSING -ftime-report  2>&1 | FileCheck %s

// Template function declarations
template <typename T>
void foo();
template <typename T, typename U>
void foo();

// Template function definitions.
template <typename T>
void foo() {}

// Template class (forward) declarations
template <typename T>
struct A;
template <typename T, typename U>
struct b;
template <typename>
struct C;
template <typename, typename>
struct D;

// Forward declarations with default parameters?
template <typename T = int>
class X1;
template <typename = int>
class X2;

// Forward declarations w/template template parameters
template <template <typename> class T>
class TTP1;
template <template <typename> class>
class TTP2;
template <template <typename X, typename Y> class T>
class TTP5;

// Forward declarations with non-type params
template <int>
class NTP0;
template <int N>
class NTP1;
template <int N = 5>
class NTP2;
template <int = 10>
class NTP3;
template <unsigned int N = 12u>
class NTP4;
template <unsigned int = 12u>
class NTP5;
template <unsigned = 15u>
class NTP6;
template <typename T, T Obj>
class NTP7;

// Template class declarations
template <typename T>
struct A {};
template <typename T, typename U>
struct B {};

namespace PR6184 {
namespace N {
template <typename T>
void bar(typename T::x);
}

template <typename T>
void N::bar(typename T::x) {}
}

// This PR occurred only in template parsing mode.
namespace PR17637 {
template <int>
struct L {
  template <typename T>
  struct O {
    template <typename U>
    static void Fun(U);
  };
};

template <int k>
template <typename T>
template <typename U>
void L<k>::O<T>::Fun(U) {}

void Instantiate() { L<0>::O<int>::Fun(0); }
}

namespace explicit_partial_specializations {
typedef char (&oneT)[1];
typedef char (&twoT)[2];
typedef char (&threeT)[3];
typedef char (&fourT)[4];
typedef char (&fiveT)[5];
typedef char (&sixT)[6];

char one[1];
char two[2];
char three[3];
char four[4];
char five[5];
char six[6];

template <bool b>
struct bool_ { typedef int type; };
template <>
struct bool_<false> {};

#define XCAT(x, y) x##y
#define CAT(x, y) XCAT(x, y)
#define sassert(_b_) bool_<(_b_)>::type CAT(var, __LINE__);

template <int>
struct L {
  template <typename T>
  struct O {
    template <typename U>
    static oneT Fun(U);
  };
};
template <int k>
template <typename T>
template <typename U>
oneT L<k>::O<T>::Fun(U) { return one; }

template <>
template <>
template <typename U>
oneT L<0>::O<char>::Fun(U) { return one; }

void Instantiate() {
  sassert(sizeof(L<0>::O<int>::Fun(0)) == sizeof(one));
  sassert(sizeof(L<0>::O<char>::Fun(0)) == sizeof(one));
}
}

template <class>
struct Foo {
  template <class _Other>
  using rebind_alloc = _Other;
};
template <class _Alloc>
struct _Wrap_alloc {
  template <class _Other>
  using rebind_alloc = typename Foo<_Alloc>::template rebind_alloc<_Other>;
  template <class>
  using rebind = _Wrap_alloc;
};
_Wrap_alloc<int>::rebind<int> w;

// CHECK: Clang time report
// CHECK:      Front end
// CHECK-NEXT: LLVM IR generation
// CHECK-NEXT: Machine code generation
// CHECK-NEXT: Optimizer
// CHECK-NEXT: Total
