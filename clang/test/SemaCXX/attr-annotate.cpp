// RUN: %clang_cc1 -std=gnu++20 -fsyntax-only -verify %s

template<bool If, typename Type>
struct enable_if {
  using type= Type;
};

template<typename Type>
struct enable_if<false, Type> {};

template<typename T1, typename T2>
struct is_same {
  static constexpr bool value = false;
};

template<typename T1>
struct is_same<T1, T1> {
  static constexpr bool value = true;
};

constexpr const char *str() {
  return "abc";
}

template<typename T, typename enable_if<!is_same<int, T>::value, int>::type = 0>
constexpr T fail_on_int(T t) {return t;}
// expected-note@-1 {{candidate template ignored: requirement}}

namespace test0 {
  template<typename T, T v>
  struct A {
    [[clang::annotate("test", fail_on_int(v))]] void t() {}
    // expected-error@-1 {{no matching function for call to 'fail_on_int'}}
    [[clang::annotate("test", (typename enable_if<!is_same<long, T>::value, int>::type)v)]] void t1() {}
    // expected-error@-1 {{failed requirement}}
  };
  A<int, 9> a;
// expected-note@-1 {{in instantiation of template class}}
  A<long, 7> a1;
// expected-note@-1 {{in instantiation of template class}}
  A<unsigned long, 6> a2;

  template<typename T>
  struct B {
    [[clang::annotate("test", ((void)T{}, 9))]] void t() {}
    // expected-error@-1 {{cannot create object of function type 'void ()'}}
  };
  B<int> b;
  B<void ()> b1;
// expected-note@-1 {{in instantiation of template class}}
}

namespace test1 {
int g_i; // expected-note {{declared here}}

[[clang::annotate("test", "arg")]] void t3() {}

template <typename T, T V>
struct B {
  static T b; // expected-note {{declared here}}
  static constexpr T cb = V;
  template <typename T1, T1 V1>
  struct foo {
    static T1 f; // expected-note {{declared here}}
    static constexpr T1 cf = V1;
    int v __attribute__((annotate("v_ann_0", str(), 90, V, g_i))) __attribute__((annotate("v_ann_1", V1)));
    // expected-error@-1 {{'annotate' attribute requires parameter 4 to be a constant expression}}
    // expected-note@-2 {{is not allowed in a constant expression}}
    [[clang::annotate("qdwqwd", cf, cb)]] void t() {}
    [[clang::annotate("qdwqwd", f, cb)]] void t1() {}
    // expected-error@-1 {{'annotate' attribute requires parameter 1 to be a constant expression}}
    // expected-note@-2 {{is not allowed in a constant expression}}
    [[clang::annotate("jui", b, cf)]] void t2() {}
    // expected-error@-1 {{'annotate' attribute requires parameter 1 to be a constant expression}}
    // expected-note@-2 {{is not allowed in a constant expression}}
    [[clang::annotate("jui", ((void)b, 0), cf)]] [[clang::annotate("jui", &b, cf, &foo::t2, str())]] void t3() {}
  };
};

static B<int long, -1>::foo<unsigned, 9> gf; // expected-note {{in instantiation of}}
static B<int long, -2> gf1;

} // namespace test1

namespace test2 {

template<int I>
int f() {
  [[clang::annotate("test", I)]] int v = 0; // expected-note {{declared here}}
  [[clang::annotate("test", v)]] int v2 = 0;
  // expected-error@-1 {{'annotate' attribute requires parameter 1 to be a constant expression}}
  // expected-note@-2 {{is not allowed in a constant expression}}
  [[clang::annotate("test", rtyui)]] int v3 = 0;
    // expected-error@-1 {{use of undeclared identifier 'rtyui'}}
}

void test() {}
}

namespace test3 {

void f() {
  int n = 10;
  int vla[n];

  [[clang::annotate("vlas are awful", sizeof(vla))]] int i = 0; // reject, the sizeof is not unevaluated
  // expected-error@-1 {{'annotate' attribute requires parameter 1 to be a constant expression}}
  // expected-note@-2 {{subexpression not valid in a constant expression}}
  [[clang::annotate("_Generic selection expression should be fine", _Generic(n, int : 0, default : 1))]]
  int j = 0; // second arg should resolve to 0 fine
}
void designator();
[[clang::annotate("function designators?", designator)]] int k = 0; // Should work?

void self() {
  [[clang::annotate("function designators?", self)]] int k = 0;
}

}

namespace test4 {
constexpr int foldable_but_invalid() {
  int *A = new int(0);
// expected-note@-1 {{allocation performed here was not deallocated}}
  return *A;
}

[[clang::annotate("", foldable_but_invalid())]] void f1() {}
// expected-error@-1 {{'annotate' attribute requires parameter 1 to be a constant expression}}

[[clang::annotate()]] void f2() {}
// expected-error@-1 {{'annotate' attribute takes at least 1 argument}}

template <typename T> [[clang::annotate()]] void f2() {}
// expected-error@-1 {{'annotate' attribute takes at least 1 argument}}
}
