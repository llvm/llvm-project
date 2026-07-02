// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++98 -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,cxx98,cxx98-20 %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,since-cxx11,cxx98-20 %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++14 -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,since-cxx11,since-cxx14,cxx98-20 %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,since-cxx11,since-cxx14,cxx98-20 %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,since-cxx11,since-cxx14,since-cxx20,cxx98-20 %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,since-cxx11,since-cxx14,since-cxx20,since-cxx23 %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++2c -fexceptions -fcxx-exceptions -pedantic-errors -verify-directives -verify=expected,since-cxx11,since-cxx14,since-cxx20,since-cxx23,since-cxx26 %s

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
#if __cplusplus >= 201103L
  using size_t = decltype(sizeof(int));

  template <class E> class initializer_list {
    const E *begin_;
    size_t size_;

  public:
#if __cplusplus >= 201402L
    constexpr
#endif
    initializer_list() : begin_(nullptr), size_(0) {}
#if __cplusplus >= 201402L
    constexpr
#endif
    initializer_list(const E *begin, size_t size)
        : begin_(begin), size_(size) {}
#if __cplusplus >= 201402L
    constexpr
#endif
    const E *begin() const { return begin_; }
#if __cplusplus >= 201402L
    constexpr
#endif
    size_t size() const { return size_; }
  };

  struct string_view {
    const char *begin_;
    constexpr string_view(const char *begin) : begin_(begin) {}
    constexpr const char *begin() const { return begin_; }
  };
#endif

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
// since-cxx20-error@-1 {{no viable constructor or deduction guide for deduction of template arguments of 'A'}}
//   since-cxx20-note@#cwg2707-A {{candidate function template not viable: requires 1 argument, but 3 were provided}}
//   since-cxx20-note@#cwg2707-A {{implicit deduction guide declared as 'template <class T, unsigned int N> A(cwg2707::A<T, N>) -> cwg2707::A<T, N>'}}
//   since-cxx20-note@#cwg2707-guide-A {{candidate template ignored: constraints not satisfied [with T = <int, int, int>]}}
//   since-cxx20-note@#cwg2707-guide-A {{because 'sizeof...(T) == 2' (3 == 2) evaluated to false}}
//   since-cxx20-note@#cwg2707-A {{candidate function template not viable: requires 0 arguments, but 3 were provided}}
//   since-cxx20-note@#cwg2707-A {{implicit deduction guide declared as 'template <class T, unsigned int N> A() -> cwg2707::A<T, N>'}}

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

namespace cwg2765 { // cwg2765: 23
static_assert(+"foo" == "foo", "");
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
//   expected-note@-2 {{comparison of addresses of potentially overlapping literals has unspecified value}}
static_assert("xfoo" + 1 == "foo\0y", "");
// expected-warning@-1 {{adding 'int' to a string does not append to the string}}
//   expected-note@-2 {{use array indexing to silence this warning}}
// expected-error@-3 {{static assertion expression is not an integral constant expression}}
//   expected-note@-4 {{comparison of addresses of potentially overlapping literals has unspecified value}}
static_assert("foo" + 0 != "bar", "");
// expected-warning@-1 {{adding 'int' to a string does not append to the string}}
//   expected-note@-2 {{use array indexing to silence this warning}}
// cxx98-error@-3 {{static assertion expression is not an integral constant expression}} FIXME
static_assert((const char*)"foo" != "oo", "");
// cxx98-error@-1 {{static assertion expression is not an integral constant expression}} FIXME

#if __cplusplus >= 201103L
constexpr const char *f() { return "foo"; }

constexpr bool b2 = f() == f();
// since-cxx11-error@-1 {{constexpr variable 'b2' must be initialized by a constant expression}}
//   since-cxx11-note@-2 {{comparison of addresses of potentially overlapping literals has unspecified value}}
constexpr const char *p = f();
constexpr bool b3 = p == p;
static_assert(b3, "");

constexpr bool b4 = &"xfoo"[1] == &"foo\0y"[0];
// since-cxx11-error@-1 {{constexpr variable 'b4' must be initialized by a constant expression}}
//   since-cxx11-note@-2 {{comparison of addresses of potentially overlapping literals has unspecified value}}
static_assert("foo" != &"bar"[0], "");
static_assert((const char *)"foo" != "oo", "");

template <class T>
constexpr bool f10(T s, T t) {
  return s.begin() == t.begin(); // #cwg2765-f10-compare
}
constexpr bool b10a = f10<std::string_view>("abc", "abc");
// since-cxx11-error@-1 {{constexpr variable 'b10a' must be initialized by a constant expression}}
//   since-cxx11-note@#cwg2765-f10-compare {{comparison of addresses of potentially overlapping literals has unspecified value}}
//   since-cxx11-note@-3 {{in call to 'f10<std::string_view>({&"abc"[0]}, {&"abc"[0]})'}}
constexpr bool b10b = f10<std::string_view>("abc", "def");
static_assert(!b10b, "");

constexpr const char *a11 = "abc";
constexpr const char *b11 = "abc";
constexpr bool f11() { return a11 == b11; } // #cwg2765-f11-compare
// cxx98-20-error@-1 {{constexpr function never produces a constant expression}}
//   cxx98-20-note@#cwg2765-f11-compare {{comparison of addresses of potentially overlapping literals has unspecified value}}
static_assert(f11() || !f11(), "");
// since-cxx11-error@-1 {{static assertion expression is not an integral constant expression}}
//   since-cxx11-note@#cwg2765-f11-compare {{comparison of addresses of potentially overlapping literals has unspecified value}}
//   since-cxx11-note@-3 {{in call to 'f11()'}}

#if __cplusplus >= 201402L
constexpr bool f(std::initializer_list<int> a, std::initializer_list<int> b) {
  return a.begin() != b.begin(); // #cwg2765-init-list-compare
}
static_assert(f({1}, {1}), "");
// since-cxx14-error@-1 {{static assertion expression is not an integral constant expression}}
//   since-cxx14-note@#cwg2765-init-list-compare {{comparison of addresses of potentially non-unique objects has unspecified value}}
//   since-cxx14-note@-3 {{in call to 'f({&{1}[0], 1}, {&{1}[0], 1})'}}

constexpr bool f9(const int *p) {
  std::initializer_list<int> il = {1, 2, 3};
  return p ? (p == il.begin()) : f9(il.begin()); // #cwg2765-f9-compare
}
constexpr bool b9 = f9(nullptr);
// since-cxx14-error@-1 {{constexpr variable 'b9' must be initialized by a constant expression}}
//   since-cxx14-note@#cwg2765-f9-compare {{comparison of addresses of potentially non-unique objects has unspecified value}}
//   since-cxx14-note@#cwg2765-f9-compare {{in call to 'f9(&{1, 2, 3}[0])'}}
//   since-cxx14-note@-4 {{in call to 'f9(nullptr)'}}

constexpr bool b10c = f10<std::initializer_list<int>>({1, 2, 3}, {1, 2, 3});
// since-cxx14-error@-1 {{constexpr variable 'b10c' must be initialized by a constant expression}}
//   since-cxx14-note@#cwg2765-f10-compare {{comparison of addresses of potentially non-unique objects has unspecified value}}
//   since-cxx14-note@-3 {{in call to 'f10<std::initializer_list<int>>({&{1, 2, 3}[0], 3}, {&{1, 2, 3}[0], 3})'}}
constexpr bool b10d = f10<std::initializer_list<int>>({1, 2, 3}, {4, 5, 6});
static_assert(!b10d, "");

constexpr bool ne(std::initializer_list<int> a,
                  std::initializer_list<int> b) {
  return a.begin() != b.begin() + 1; // #cwg2765-annex-c-compare
}
constexpr bool annex_c = ne({2, 3}, {1, 2, 3});
// since-cxx14-error@-1 {{constexpr variable 'annex_c' must be initialized by a constant expression}}
//   since-cxx14-note@#cwg2765-annex-c-compare {{comparison of addresses of potentially non-unique objects has unspecified value}}
//   since-cxx14-note@-3 {{in call to 'ne({&{2, 3}[0], 2}, {&{1, 2, 3}[0], 3})'}}

int a_order[10];
constexpr int inc(int &i) { return (i += 1); }
constexpr int twox(int &i) { return (i *= 2); }
constexpr int f_order(int i) { return inc(i) + twox(i); }
constexpr bool g_order() { return &a_order[f_order(1)] == &a_order[6]; }
constexpr bool b_order = g_order();
static_assert(b_order, "");

// Aggregate carrying a std::initializer_list member: the backing array's
// extending declaration is the enclosing aggregate, not the
// initializer_list. The precise marker on MaterializeTemporaryExpr must
// still recognise the backing array.
struct WithIL { std::initializer_list<int> il; };
constexpr WithIL agg_a{{1}}, agg_b{{1}};
constexpr bool agg_same = agg_a.il.begin() == agg_b.il.begin();
// since-cxx14-error@-1 {{constexpr variable 'agg_same' must be initialized by a constant expression}}
//   since-cxx14-note@-2 {{comparison of addresses of potentially non-unique objects has unspecified value}}

constexpr WithIL agg_c{{1}}, agg_d{{2}};
constexpr bool agg_different = agg_c.il.begin() == agg_d.il.begin();
static_assert(!agg_different, "");

// Rvalue reference to array bound to a braced-init-list: the materialized
// array is not a std::initializer_list backing array, so address
// comparisons across two such temporaries are well-defined.
constexpr bool rvref_arr_same(const int (&&a)[3], const int (&&b)[3]) {
  return &a[0] == &b[0];
}
constexpr bool rvref_ok = rvref_arr_same({1, 2, 3}, {1, 2, 3});
static_assert(!rvref_ok, "");
#endif
#endif
} // namespace cwg2765

namespace cwg2770 { // cwg2770: 20 open 2023-07-14
#if __cplusplus >= 202002L
template<typename T>
struct B {
  static_assert(sizeof(T) == 1);
  using type = int;
};

template<typename T>
int f(T t, typename B<T>::type u) requires (sizeof(t) == 1);

template<typename T>
int f(T t, long);

int i = f(1, 2);
int j = f('a', 2);

#endif
} // namespace cwg2770

namespace cwg2780 { // cwg2780: 2.7

void f();

void g() {
  (void)reinterpret_cast<void(&)(int)>(f);
}

} // namespace cwg2780

namespace cwg2785 { // cwg2785: 10
#if __cplusplus >= 202002L
void g(void *); // #cwg2785-g

template <typename T>
void f() {
  g(requires { T(); });
  // since-cxx20-error@-1 {{no matching function for call to 'g'}}
  //   since-cxx20-note@#cwg2785-g {{candidate function not viable: no known conversion from 'bool' to 'void *' for 1st argument}}
}
#endif
} // namespace cwg2785

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
