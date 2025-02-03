// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++98 -pedantic-errors -verify=expected,cxx98 %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++14 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -pedantic-errors -verify=expected,since-cxx20 %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -pedantic-errors -verify=expected,since-cxx20,since-cxx23 %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++2c -pedantic-errors -verify=expected,since-cxx20,since-cxx23,since-cxx26 %s

#if __cplusplus == 199711L
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__)
// cxx98-error@-1 {{variadic macros are a C99 feature}}
#endif

#if __cplusplus == 199711L
#define __enable_constant_folding(x) (__builtin_constant_p(x) ? (x) : (x))
#else
#define __enable_constant_folding
#endif

namespace std {
#if __cplusplus >= 202002L
  struct strong_ordering {
    int n;
    constexpr operator int() const { return n; }
    static const strong_ordering less, equal, greater;
  };
  constexpr strong_ordering strong_ordering::less{-1},
      strong_ordering::equal{0}, strong_ordering::greater{1};
#endif
} // namespace std

namespace cwg2707 { // cwg2707: 20

#if __cplusplus >= 202002L

template <class T, unsigned N> struct A { // #cwg2707-A
  T value[N];
};

template <typename... T>
A(T...) -> A<int, sizeof...(T)> requires (sizeof...(T) == 2); // #cwg2707-guide-A

// Brace elision is not allowed for synthesized CTAD guides if the array size
// is value-dependent.
// So this should pick up our explicit deduction guide.
A a = {1, 2};

A b = {3, 4, 5};
// since-cxx20-error@-1 {{no viable constructor or deduction guide}}
//   since-cxx20-note@#cwg2707-A {{candidate function template not viable}}
//   since-cxx20-note@#cwg2707-A {{implicit deduction guide}}
//   since-cxx20-note@#cwg2707-guide-A {{constraints not satisfied}}
//   since-cxx20-note@#cwg2707-guide-A {{because 'sizeof...(T) == 2' (3 == 2) evaluated to false}}
//   since-cxx20-note@#cwg2707-A {{candidate function template not viable}}
//   since-cxx20-note@#cwg2707-A {{implicit deduction guide}}

#endif

} // namespace cwg2707

namespace cwg2718 { // cwg2718: 2.7
struct B {};
struct D;

void f(B b) {
  static_cast<D&>(b);
  // expected-error@-1 {{non-const lvalue reference to type 'D' cannot bind to a value of unrelated type 'B'}}
}

struct D : B {};
} // namespace cwg2718

namespace cwg2749 { // cwg2749: 20

extern int x[2];
struct Y {
  int i;
  int j;
};
extern Y y[2];

static_assert(__enable_constant_folding(static_cast<void*>(&x[0]) < static_cast<void*>(&x[1])), "");
static_assert(__enable_constant_folding(static_cast<void*>(&y[0].i) < static_cast<void*>(&y[0].j)), "");
static_assert(__enable_constant_folding(static_cast<void*>(&y[0].j) < static_cast<void*>(&y[1].i)), "");

#if __cplusplus >= 202002L
static_assert((static_cast<void*>(&x[0]) <=> static_cast<void*>(&x[1])) == std::strong_ordering::less);
static_assert((static_cast<void*>(&y[0].i) <=> static_cast<void*>(&y[0].j)) == std::strong_ordering::less);
static_assert((static_cast<void*>(&y[0].j) <=> static_cast<void*>(&y[1].i)) == std::strong_ordering::less);
#endif

} // namespace cwg2749

namespace cwg2759 { // cwg2759: 19
#if __cplusplus >= 201103L

struct CStruct {
  int one;
  int two;
};

struct CEmptyStruct {};
struct CEmptyStruct2 {};

struct CStructNoUniqueAddress {
  int one;
  [[no_unique_address]] int two;
};

struct CStructNoUniqueAddress2 {
  int one;
  [[no_unique_address]] int two;
};

union UnionLayout {
  int a;
  double b;
  CStruct c;
  [[no_unique_address]] CEmptyStruct d;
  [[no_unique_address]] CEmptyStruct2 e;
};

union UnionLayout2 {
  CStruct c;
  int a;
  CEmptyStruct2 e;
  double b;
  [[no_unique_address]] CEmptyStruct d;
};

union UnionLayout3 {
  CStruct c;
  int a;
  double b;
  [[no_unique_address]] CEmptyStruct d;
};

struct StructWithAnonUnion {
  union {
    int a;
    double b;
    CStruct c;
    [[no_unique_address]] CEmptyStruct d;
    [[no_unique_address]] CEmptyStruct2 e;
  };
};

struct StructWithAnonUnion2 {
  union {
    CStruct c;
    int a;
    CEmptyStruct2 e;
    double b;
    [[no_unique_address]] CEmptyStruct d;
  };
};

struct StructWithAnonUnion3 {
  union {
    CStruct c;
    int a;
    CEmptyStruct2 e;
    double b;
    [[no_unique_address]] CEmptyStruct d;
  } u;
};

static_assert(__is_layout_compatible(CStruct, CStructNoUniqueAddress) != bool(__has_cpp_attribute(no_unique_address)), "");
static_assert(__is_layout_compatible(CStructNoUniqueAddress, CStructNoUniqueAddress2) != bool(__has_cpp_attribute(no_unique_address)), "");
static_assert(!__is_layout_compatible(UnionLayout, UnionLayout2), "");
static_assert(!__is_layout_compatible(UnionLayout, UnionLayout3), "");
static_assert(!__is_layout_compatible(StructWithAnonUnion, StructWithAnonUnion2), "");
static_assert(!__is_layout_compatible(StructWithAnonUnion, StructWithAnonUnion3), "");
#endif
} // namespace cwg2759

namespace cwg2789 { // cwg2789: 18
#if __cplusplus >= 202302L
template <typename T = int>
struct Base {
    constexpr void g(); // #cwg2789-g1
};

template <typename T = int>
struct Base2 {
    constexpr void g() requires true;  // #cwg2789-g2
};

template <typename T = int>
struct S : Base<T>, Base2<T> {
    constexpr void f();
    constexpr void f(this S&) requires true{};

    using Base<T>::g;
    using Base2<T>::g;
};

void test() {
    S<> s;
    s.f();
    s.g();
    // since-cxx23-error@-1 {{call to member function 'g' is ambiguous}}
    //   since-cxx23-note@#cwg2789-g1 {{candidate function}}
    //   since-cxx23-note@#cwg2789-g2 {{candidate function}}
}
#endif
} // namespace cwg2789

namespace cwg2798 { // cwg2798: 17
#if __cplusplus > 202302L
struct string {
  constexpr string() {
    data_ = new char[6]();
    __builtin_memcpy(data_, "Hello", 5);
    data_[5] = 0;
  }
  constexpr ~string() { delete[] data_; }
  constexpr unsigned long size() const { return 5; };
  constexpr const char *data() const { return data_; }

  char *data_;
};
struct X {
  string s;
};
consteval X f() { return {}; }

static_assert(false, f().s);
// since-cxx26-error@-1 {{static assertion failed: Hello}}
#endif
} // namespace cwg2798
