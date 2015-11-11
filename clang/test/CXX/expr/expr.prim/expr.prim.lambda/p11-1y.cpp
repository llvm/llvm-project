// RUN: %clang_cc1 -std=c++1y %s -verify

const char *has_no_member = [x("hello")] {}.x; // expected-error {{no member named 'x'}}

double f;
auto with_float = [f(1.0f)] {
  using T = decltype(f);
  using T = float;
};
auto with_float_2 = [&f(f)] { // ok, refers to outer f
  using T = decltype(f);
  using T = double&;
};

// Within the lambda-expression's compound-statement,
// the identifier in the init-capture hides any declaration
// of the same name in scopes enclosing the lambda-expression.
void hiding() {
  char c;
  (void) [c("foo")] {
    static_assert(sizeof(c) == sizeof(const char*), "");
  };
  (void) [c("bar")] () -> decltype(c) { // outer c, not init-capture
    return "baz"; // expected-error {{cannot initialize}}
  };
}

struct ExplicitCopy {
  ExplicitCopy(); // expected-note 2{{not viable}}
  explicit ExplicitCopy(const ExplicitCopy&);
};
auto init_kind_1 = [ec(ExplicitCopy())] {};
auto init_kind_2 = [ec = ExplicitCopy()] {}; // expected-error {{no matching constructor}}

template<typename T> void init_kind_template() {
  auto init_kind_1 = [ec(T())] {};
  auto init_kind_2 = [ec = T()] {}; // expected-error {{no matching constructor}}
}
template void init_kind_template<int>();
template void init_kind_template<ExplicitCopy>(); // expected-note {{instantiation of}}

void void_fn();
int overload_fn();
int overload_fn(int);

auto bad_init_1 = [a()] {}; // expected-error {{expected expression}}
auto bad_init_2 = [a(1, 2)] {}; // expected-error {{initializer for lambda capture 'a' contains multiple expressions}}
auto bad_init_3 = [&a(void_fn())] {}; // expected-error {{cannot form a reference to 'void'}}
auto bad_init_4 = [a(void_fn())] {}; // expected-error {{has incomplete type 'void'}}
auto bad_init_5 = [a(overload_fn)] {}; // expected-error {{cannot deduce type for lambda capture 'a' from initializer of type '<overloaded function}}
auto bad_init_6 = [a{overload_fn}] {}; // expected-error {{cannot deduce type for lambda capture 'a' from initializer list}}
auto bad_init_7 = [a{{1}}] {}; // expected-error {{cannot deduce type for lambda capture 'a' from nested initializer list}}

template<typename...T> void pack_1(T...t) { (void)[a(t...)] {}; } // expected-error {{initializer missing for lambda capture 'a'}}
template void pack_1<>(); // expected-note {{instantiation of}}

// FIXME: Might need lifetime extension for the temporary here.
// See DR1695.
auto a = [a(4), b = 5, &c = static_cast<const int&&>(0)] {
  static_assert(sizeof(a) == sizeof(int), "");
  static_assert(sizeof(b) == sizeof(int), "");
  using T = decltype(c);
  using T = const int &;
};
auto b = [a{0}] {}; // OK, per N3922

struct S { S(); S(S&&); };
template<typename T> struct remove_reference { typedef T type; };
template<typename T> struct remove_reference<T&> { typedef T type; };
template<typename T> decltype(auto) move(T &&t) { return static_cast<typename remove_reference<T>::type&&>(t); }
auto s = [s(move(S()))] {};

template<typename T> T instantiate_test(T t) {
  [x(&t)]() { *x = 1; } (); // expected-error {{assigning to 'const char *'}}
  return t;
}
int instantiate_test_1 = instantiate_test(0);
const char *instantiate_test_2 = instantiate_test("foo"); // expected-note {{here}}
