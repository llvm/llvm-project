// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11 -triple=x86_64-linux-gnu

int f(); // expected-note {{declared here}}

static_assert(f(), "f"); // expected-error {{static_assert expression is not an integral constant expression}} expected-note {{non-constexpr function 'f' cannot be used in a constant expression}}
static_assert(true, "true is not false");
static_assert(false, "false is false"); // expected-error {{static_assert failed "false is false"}}

void g() {
    static_assert(false, "false is false"); // expected-error {{static_assert failed "false is false"}}
}

class C {
    static_assert(false, "false is false"); // expected-error {{static_assert failed "false is false"}}
};

template<int N> struct T {
    static_assert(N == 2, "N is not 2!"); // expected-error {{static_assert failed due to requirement '1 == 2' "N is not 2!"}}
};

T<1> t1; // expected-note {{in instantiation of template class 'T<1>' requested here}}
T<2> t2;

template<typename T> struct S {
    static_assert(sizeof(T) > sizeof(char), "Type not big enough!"); // expected-error {{static_assert failed due to requirement 'sizeof(char) > sizeof(char)' "Type not big enough!"}}
};

S<char> s1; // expected-note {{in instantiation of template class 'S<char>' requested here}}
S<int> s2;

static_assert(false, L"\xFFFFFFFF"); // expected-error {{static_assert failed L"\xFFFFFFFF"}}
static_assert(false, u"\U000317FF"); // expected-error {{static_assert failed u"\U000317FF"}}
// FIXME: render this as u8"\u03A9"
static_assert(false, u8"Ω"); // expected-error {{static_assert failed u8"\316\251"}}
static_assert(false, L"\u1234"); // expected-error {{static_assert failed L"\x1234"}}
static_assert(false, L"\x1ff" "0\x123" "fx\xfffff" "goop"); // expected-error {{static_assert failed L"\x1FF""0\x123""fx\xFFFFFgoop"}}

template<typename T> struct AlwaysFails {
  // Only give one error here.
  static_assert(false, ""); // expected-error {{static_assert failed}}
};
AlwaysFails<int> alwaysFails;

template<typename T> struct StaticAssertProtected {
  static_assert(__is_literal(T), ""); // expected-error {{static_assert failed}}
  static constexpr T t = {}; // no error here
};
struct X { ~X(); };
StaticAssertProtected<int> sap1;
StaticAssertProtected<X> sap2; // expected-note {{instantiation}}

static_assert(true); // expected-warning {{C++17 extension}}
static_assert(false); // expected-error-re {{failed{{$}}}} expected-warning {{extension}}


// Diagnostics for static_assert with multiple conditions
template<typename T> struct first_trait {
  static const bool value = false;
};

template<>
struct first_trait<X> {
  static const bool value = true;
};

template<typename T> struct second_trait {
  static const bool value = false;
};

static_assert(first_trait<X>::value && second_trait<X>::value, "message"); // expected-error{{static_assert failed due to requirement 'second_trait<X>::value' "message"}}

namespace std {

template <class Tp, Tp v>
struct integral_constant {
  static const Tp value = v;
  typedef Tp value_type;
  typedef integral_constant type;
  constexpr operator value_type() const noexcept { return value; }
  constexpr value_type operator()() const noexcept { return value; }
};

template <class Tp, Tp v>
const Tp integral_constant<Tp, v>::value;

typedef integral_constant<bool, true> true_type;
typedef integral_constant<bool, false> false_type;

template <class Tp>
struct is_const : public false_type {};
template <class Tp>
struct is_const<Tp const> : public true_type {};

// We do not define is_same in terms of integral_constant to check that both implementations are supported.
template <typename T, typename U>
struct is_same {
  static const bool value = false;
};

template <typename T>
struct is_same<T, T> {
  static const bool value = true;
};

} // namespace std

struct ExampleTypes {
  explicit ExampleTypes(int);
  using T = int;
  using U = float;
};

static_assert(std::is_same<ExampleTypes::T, ExampleTypes::U>::value, "message");
// expected-error@-1{{static_assert failed due to requirement 'std::is_same<int, float>::value' "message"}}
static_assert(std::is_const<ExampleTypes::T>::value, "message");
// expected-error@-1{{static_assert failed due to requirement 'std::is_const<int>::value' "message"}}
static_assert(!std::is_const<const ExampleTypes::T>::value, "message");
// expected-error@-1{{static_assert failed due to requirement '!std::is_const<const int>::value' "message"}}
static_assert(!(std::is_const<const ExampleTypes::T>::value), "message");
// expected-error@-1{{static_assert failed due to requirement '!(std::is_const<const int>::value)' "message"}}
static_assert(std::is_const<const ExampleTypes::T>::value == false, "message");
// expected-error@-1{{static_assert failed due to requirement 'std::is_const<const int>::value == false' "message"}}
static_assert(!(std::is_const<const ExampleTypes::T>::value == true), "message");
// expected-error@-1{{static_assert failed due to requirement '!(std::is_const<const int>::value == true)' "message"}}
static_assert(std::is_const<ExampleTypes::T>(), "message");
// expected-error@-1{{static_assert failed due to requirement 'std::is_const<int>()' "message"}}
static_assert(!(std::is_const<const ExampleTypes::T>()()), "message");
// expected-error@-1{{static_assert failed due to requirement '!(std::is_const<const int>()())' "message"}}
static_assert(std::is_same<decltype(std::is_const<const ExampleTypes::T>()), int>::value, "message");
// expected-error@-1{{static_assert failed due to requirement 'std::is_same<std::is_const<const int>, int>::value' "message"}}
static_assert(std::is_const<decltype(ExampleTypes::T(3))>::value, "message");
// expected-error@-1{{static_assert failed due to requirement 'std::is_const<int>::value' "message"}}
static_assert(std::is_const<decltype(ExampleTypes::T())>::value, "message");
// expected-error@-1{{static_assert failed due to requirement 'std::is_const<int>::value' "message"}}
static_assert(std::is_const<decltype(ExampleTypes(3))>::value, "message");
// expected-error@-1{{static_assert failed due to requirement 'std::is_const<ExampleTypes>::value' "message"}}

struct BI_tag {};
struct RAI_tag : BI_tag {};
struct MyIterator {
  using tag = BI_tag;
};
struct MyContainer {
  using iterator = MyIterator;
};
template <class Container>
void foo() {
  static_assert(std::is_same<RAI_tag, typename Container::iterator::tag>::value, "message");
  // expected-error@-1{{static_assert failed due to requirement 'std::is_same<RAI_tag, BI_tag>::value' "message"}}
}
template void foo<MyContainer>();
// expected-note@-1{{in instantiation of function template specialization 'foo<MyContainer>' requested here}}

namespace ns {
template <typename T, int v>
struct NestedTemplates1 {
  struct NestedTemplates2 {
    template <typename U>
    struct NestedTemplates3 : public std::is_same<T, U> {};
  };
};
} // namespace ns

template <typename T, typename U, int a>
void foo2() {
  static_assert(::ns::NestedTemplates1<T, a>::NestedTemplates2::template NestedTemplates3<U>::value, "message");
  // expected-error@-1{{static_assert failed due to requirement '::ns::NestedTemplates1<int, 3>::NestedTemplates2::NestedTemplates3<float>::value' "message"}}
}
template void foo2<int, float, 3>();
// expected-note@-1{{in instantiation of function template specialization 'foo2<int, float, 3>' requested here}}

template <class T>
void foo3(T t) {
  static_assert(std::is_const<T>::value, "message");
  // expected-error-re@-1{{static_assert failed due to requirement 'std::is_const<(lambda at {{.*}}static-assert.cpp:{{[0-9]*}}:{{[0-9]*}})>::value' "message"}}
  static_assert(std::is_const<decltype(t)>::value, "message");
  // expected-error-re@-1{{static_assert failed due to requirement 'std::is_const<(lambda at {{.*}}static-assert.cpp:{{[0-9]*}}:{{[0-9]*}})>::value' "message"}}
}
void callFoo3() {
  foo3([]() {});
  // expected-note@-1{{in instantiation of function template specialization 'foo3<(lambda at }}
}

template <class T>
void foo4(T t) {
  static_assert(std::is_const<typename T::iterator>::value, "message");
  // expected-error@-1{{type 'int' cannot be used prior to '::' because it has no members}}
}
void callFoo4() { foo4(42); }
// expected-note@-1{{in instantiation of function template specialization 'foo4<int>' requested here}}
