// RUN: %clang_cc1 -std=c++11 -ast-print %s | FileCheck %s

// CHECK: auto operator""_foo(const char *p, decltype(sizeof(int))) -> decltype(nullptr);
auto operator"" _foo(const char *p, decltype(sizeof(int))) -> decltype(nullptr);

// CHECK: decltype(""_foo) operator""_bar(unsigned long long);
decltype(""_foo) operator"" _bar(unsigned long long);

// CHECK: decltype(42_bar) operator""_baz(long double);
decltype(42_bar) operator"" _baz(long double);

// CHECK: decltype(4.5_baz) operator""_baz(char);
decltype(4.5_baz) operator"" _baz(char);

// CHECK: const char *operator""_quux(const char *);
const char *operator"" _quux(const char *);

// CHECK: template <char ...> const char *operator""_fritz();
template<char...> const char *operator"" _fritz();

// CHECK: const char *p1 = "bar1"_foo;
const char *p1 = "bar1"_foo;
// CHECK: const char *p2 = "bar2"_foo;
const char *p2 = R"x(bar2)x"_foo;
// CHECK: const char *p3 = u8"bar3"_foo;
const char *p3 = u8"bar3"_foo;
// CHECK: const char *p4 = 297_bar;
const char *p4 = 0x129_bar;
// CHECK: const char *p5 = 1.0E+12_baz;
const char *p5 = 1e12_baz;
// CHECK: const char *p6 = 'x'_baz;
const char *p6 = 'x'_baz;
// CHECK: const char *p7 = 123_quux;
const char *p7 = 123_quux;
// CHECK: const char *p8 = 4.9_quux;
const char *p8 = 4.9_quux;
// CHECK: const char *p9 = 0x42e3F_fritz;
const char *p9 = 0x42e3F_fritz;
// CHECK: const char *p10 = 3.300e+15_fritz;
const char *p10 = 3.300e+15_fritz;

template <class C, C...> const char *operator"" _suffix();
// CHECK: const char *PR23120 = operator""_suffix<char32_t, 66615>();
const char *PR23120 = U"𐐷"_suffix;

// PR28885
struct A {
  A();
};
struct B : A {
  using A::A; // CHECK:      using A::A;
};            // CHECK-NEXT: };

// CHECK: ;
;
// CHECK-NOT: ;

void g(int n) {
  int buffer[n];     // CHECK: int buffer[n];
  [&buffer]() {}();  // CHECK: [&buffer]
}

