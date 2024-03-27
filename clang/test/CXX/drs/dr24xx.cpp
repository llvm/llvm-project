// RUN: %clang_cc1 -std=c++98 %s -verify=expected
// RUN: %clang_cc1 -std=c++11 %s -verify=expected
// RUN: %clang_cc1 -std=c++14 %s -verify=expected
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,since-cxx17
// RUN: %clang_cc1 -std=c++20 %s -verify=expected,since-cxx17
// RUN: %clang_cc1 -std=c++23 %s -verify=expected,since-cxx17
// RUN: %clang_cc1 -std=c++2c %s -verify=expected,since-cxx17

#if __cplusplus <= 201402L
// expected-no-diagnostics
#endif

namespace dr2406 { // dr2406: 5
#if __cplusplus >= 201703L
void fallthrough(int n) {
  void g(), h(), i();
  switch (n) {
  case 1:
  case 2:
    g();
    [[fallthrough]];
  case 3: // warning on fallthrough discouraged
    do {
      [[fallthrough]];
      // since-cxx17-error@-1 {{fallthrough annotation does not directly precede switch label}}
    } while (false);
  case 6:
    do {
      [[fallthrough]];
      // since-cxx17-error@-1 {{fallthrough annotation does not directly precede switch label}}
    } while (n);
  case 7:
    while (false) {
      [[fallthrough]];
      // since-cxx17-error@-1 {{fallthrough annotation does not directly precede switch label}}
    }
  case 5:
    h();
  case 4: // implementation may warn on fallthrough
    i();
    [[fallthrough]];
    // since-cxx17-error@-1 {{fallthrough annotation does not directly precede switch label}}
  }
}
#endif
}

namespace dr2450 { // dr2450: 18 review P2308R1
#if __cplusplus >= 202302L
struct S {int a;};
template <S s>
void f(){}

void test() {
f<{0}>();
f<{.a= 0}>();
}

#endif
}

namespace dr2459 { // dr2459: 18 drafting P2308R1
#if __cplusplus >= 202302L
struct A {
  constexpr A(float) {}
};
template<A> struct X {};
X<1> x;
#endif
}

namespace dr2445 { // dr2445: 19
#if __cplusplus >= 202002L
  template <typename> constexpr bool F = false;
  template <typename T> struct A { };

  template <typename T, typename U>
  bool operator==(T, A<U *>);

  template <typename T, typename U>
  bool operator!=(A<T>, U) {
   static_assert(F<T>, "Isn't this less specialized?");
   return false;
  }

  bool f(A<int> ax, A<int *> ay) { return ay != ax; }

  template<class T> concept AlwaysTrue=true;
  template <class T> struct B {
    template <AlwaysTrue U>
    bool operator==(const B<U>&)const;
  };


  template <typename U>
  bool operator==(const B<int>&,const B<U>&) {
   static_assert(F<int>, "Isn't this less specialized?");
   return false;
  }

  bool g(B<int> bx, B<int *> by) { return bx == by; }

  struct C{
    template<AlwaysTrue T>
    int operator+(T){return 0;}
    template<class T>
    void operator-(T){}
  };
  template<class T>
  void operator+(C&&,T){}
  template<AlwaysTrue T>
  int operator-(C&&,T){return 0;}

  void t(int* iptr){
    int x1 = C{} + iptr;
    int x2 = C{} - iptr;
  }

  struct D{
    template<AlwaysTrue T>
    int operator+(T) volatile {return 1;}
  };

  template<class T>
  void operator+(volatile D&,T) {}

  int foo(volatile D& d){
    return d + 1;
  }
#endif
}
