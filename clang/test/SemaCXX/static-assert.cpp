// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11 -pedantic -triple=x86_64-linux-gnu -Wno-invalid-utf8

int f(); // expected-note {{declared here}}

static_assert(f(), "f"); // expected-error {{static assertion expression is not an integral constant expression}} expected-note {{non-constexpr function 'f' cannot be used in a constant expression}}
static_assert(true, "true is not false");
static_assert(false, "false is false"); // expected-error {{static assertion failed: false is false}}

void g() {
    static_assert(false, "false is false"); // expected-error {{static assertion failed: false is false}}
}

class C {
    static_assert(false, "false is false"); // expected-error {{static assertion failed: false is false}}
};

template<int N> struct T {
    static_assert(N == 2, "N is not 2!"); // expected-error {{static assertion failed due to requirement '1 == 2': N is not 2!}}
};

T<1> t1; // expected-note {{in instantiation of template class 'T<1>' requested here}}
T<2> t2;

template<typename T> struct S {
    static_assert(sizeof(T) > sizeof(char), "Type not big enough!"); // expected-error {{static assertion failed due to requirement 'sizeof(char) > sizeof(char)': Type not big enough!}} \
                                                                     // expected-note {{1 > 1}}
};

S<char> s1; // expected-note {{in instantiation of template class 'S<char>' requested here}}
S<int> s2;

static_assert(false, L"\xFFFFFFFF"); // expected-error {{static assertion failed: L"\xFFFFFFFF"}}
static_assert(false, u"\U000317FF"); // expected-error {{static assertion failed: u"\U000317FF"}}

static_assert(false, u8"Ω"); // expected-error {{static assertion failed: u8"\316\251"}}
static_assert(false, L"\u1234"); // expected-error {{static assertion failed: L"\x1234"}}
static_assert(false, L"\x1ff" "0\x123" "fx\xfffff" "goop"); // expected-error {{static assertion failed: L"\x1FF""0\x123""fx\xFFFFFgoop"}}

static_assert(false, R"(a
\tb
c
)"); // expected-error@-3 {{static assertion failed: a\n\tb\nc\n}}

static_assert(false, "\u0080\u0081\u0082\u0083\u0099\u009A\u009B\u009C\u009D\u009E\u009F");
// expected-error@-1 {{static assertion failed: <U+0080><U+0081><U+0082><U+0083><U+0099><U+009A><U+009B><U+009C><U+009D><U+009E><U+009F>}}

//! Contains RTL/LTR marks
static_assert(false, "\u200Eabc\u200Fdef\u200Fgh"); // expected-error {{static assertion failed: ‎abc‏def‏gh}}

//! Contains ZWJ/regional indicators
static_assert(false, "🏳️‍🌈 🏴󠁧󠁢󠁥󠁮󠁧󠁿 🇪🇺"); // expected-error {{static assertion failed: 🏳️‍🌈 🏴󠁧󠁢󠁥󠁮󠁧󠁿 🇪🇺}}

template<typename T> struct AlwaysFails {
  // Only give one error here.
  static_assert(false, ""); // expected-error {{static assertion failed}}
};
AlwaysFails<int> alwaysFails;

template<typename T> struct StaticAssertProtected {
  static_assert(__is_literal(T), ""); // expected-error {{static assertion failed}}
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

static_assert(first_trait<X>::value && second_trait<X>::value, "message"); // expected-error{{static assertion failed due to requirement 'second_trait<X>::value': message}}

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
// expected-error@-1{{static assertion failed due to requirement 'std::is_same<int, float>::value': message}}
static_assert(std::is_const<ExampleTypes::T>::value, "message");
// expected-error@-1{{static assertion failed due to requirement 'std::is_const<int>::value': message}}
static_assert(!std::is_const<const ExampleTypes::T>::value, "message");
// expected-error@-1{{static assertion failed due to requirement '!std::is_const<const int>::value': message}}
static_assert(!(std::is_const<const ExampleTypes::T>::value), "message");
// expected-error@-1{{static assertion failed due to requirement '!(std::is_const<const int>::value)': message}}
static_assert(std::is_const<const ExampleTypes::T>::value == false, "message");
// expected-error@-1{{static assertion failed due to requirement 'std::is_const<const int>::value == false': message}}
static_assert(!(std::is_const<const ExampleTypes::T>::value == true), "message");
// expected-error@-1{{static assertion failed due to requirement '!(std::is_const<const int>::value == true)': message}}
static_assert(std::is_const<ExampleTypes::T>(), "message");
// expected-error@-1{{static assertion failed due to requirement 'std::is_const<int>()': message}}
static_assert(!(std::is_const<const ExampleTypes::T>()()), "message");
// expected-error@-1{{static assertion failed due to requirement '!(std::is_const<const int>()())': message}}
static_assert(std::is_same<decltype(std::is_const<const ExampleTypes::T>()), int>::value, "message");
// expected-error@-1{{static assertion failed due to requirement 'std::is_same<std::is_const<const int>, int>::value': message}}
static_assert(std::is_const<decltype(ExampleTypes::T(3))>::value, "message");
// expected-error@-1{{static assertion failed due to requirement 'std::is_const<int>::value': message}}
static_assert(std::is_const<decltype(ExampleTypes::T())>::value, "message");
// expected-error@-1{{static assertion failed due to requirement 'std::is_const<int>::value': message}}
static_assert(std::is_const<decltype(ExampleTypes(3))>::value, "message");
// expected-error@-1{{static assertion failed due to requirement 'std::is_const<ExampleTypes>::value': message}}

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
  // expected-error@-1{{static assertion failed due to requirement 'std::is_same<RAI_tag, BI_tag>::value': message}}
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
  // expected-error@-1{{static assertion failed due to requirement '::ns::NestedTemplates1<int, 3>::NestedTemplates2::NestedTemplates3<float>::value': message}}
}
template void foo2<int, float, 3>();
// expected-note@-1{{in instantiation of function template specialization 'foo2<int, float, 3>' requested here}}

template <class T>
void foo3(T t) {
  static_assert(std::is_const<T>::value, "message");
  // expected-error-re@-1{{static assertion failed due to requirement 'std::is_const<(lambda at {{.*}}static-assert.cpp:{{[0-9]*}}:{{[0-9]*}})>::value': message}}
  static_assert(std::is_const<decltype(t)>::value, "message");
  // expected-error-re@-1{{static assertion failed due to requirement 'std::is_const<(lambda at {{.*}}static-assert.cpp:{{[0-9]*}}:{{[0-9]*}})>::value': message}}
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

static_assert(42, "message");
static_assert(42.0, "message"); // expected-warning {{implicit conversion from 'double' to 'bool' changes value from 42 to true}}
constexpr int *p = 0;
static_assert(p, "message"); // expected-error {{static assertion failed}}

struct NotBool {
} notBool;
constexpr NotBool constexprNotBool;
static_assert(notBool, "message");          // expected-error {{value of type 'struct NotBool' is not contextually convertible to 'bool'}}
static_assert(constexprNotBool, "message"); // expected-error {{value of type 'const NotBool' is not contextually convertible to 'bool'}}

static_assert(1 , "") // expected-error {{expected ';' after 'static_assert'}}


namespace Diagnostics {
  /// No notes for literals.
  static_assert(false, ""); // expected-error {{failed}}
  static_assert(1.0 > 2.0, ""); // expected-error {{failed}}
  static_assert('c' == 'd', ""); // expected-error {{failed}}
  static_assert(1 == 2, ""); // expected-error {{failed}}

  /// Simple things are ignored.
  static_assert(1 == (-(1)), ""); //expected-error {{failed}}

  /// Chars are printed as chars.
  constexpr char getChar() {
    return 'c';
  }
  static_assert(getChar() == 'a', ""); // expected-error {{failed}} \
                                       // expected-note {{evaluates to ''c' == 'a''}}

  /// Bools are printed as bools.
  constexpr bool invert(bool b) {
    return !b;
  }
  static_assert(invert(true) == invert(false), ""); // expected-error {{failed}} \
                                                    // expected-note {{evaluates to 'false == true'}}

  /// No notes here since we compare a bool expression with a bool literal.
  static_assert(invert(true) == true, ""); // expected-error {{failed}}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
  constexpr _Complex float com = {5,6};
  constexpr _Complex float com2 = {1, 9};
  static_assert(com == com2, ""); // expected-error {{failed}} \
                                  // expected-note {{evaluates to '(5 + 6i) == (1 + 9i)'}}
#pragma clang diagnostic pop

#define CHECK_4(x) ((x) == 4)
#define A_IS_B (a == b)
  static_assert(CHECK_4(5), ""); // expected-error {{failed}}

  constexpr int a = 4;
  constexpr int b = 5;
  static_assert(CHECK_4(a) && A_IS_B, ""); // expected-error {{failed}} \
                                           // expected-note {{evaluates to '4 == 5'}}


}
