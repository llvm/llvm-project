// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

int i __attribute__((flatten)); // expected-error {{'flatten' attribute only applies to functions}}

void f1() __attribute__((flatten));
void f2() __attribute__((flatten(1))); // expected-error {{'flatten' attribute takes no arguments}}

template <typename T>
void tf1() __attribute__((flatten));

int f3(int __attribute__((flatten)), int); // expected-error{{'flatten' attribute only applies to functions}}

struct A {
  int f __attribute__((flatten));  // expected-error{{'flatten' attribute only applies to functions}}
  void mf1() __attribute__((flatten));
  static void mf2() __attribute__((flatten));
};

int ci [[gnu::flatten]]; // expected-error {{'flatten' attribute only applies to functions}}

[[gnu::flatten]] void cf1();
[[gnu::flatten(1)]] void cf2(); // expected-error {{'flatten' attribute takes no arguments}}

template <typename T>
[[gnu::flatten]]
void ctf1();

int cf3(int c[[gnu::flatten]], int); // expected-error{{'flatten' attribute only applies to functions}}

struct CA {
  int f [[gnu::flatten]];  // expected-error{{'flatten' attribute only applies to functions}}
  [[gnu::flatten]] void mf1();
  [[gnu::flatten]] static void mf2();
};
