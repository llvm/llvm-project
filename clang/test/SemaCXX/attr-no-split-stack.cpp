// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

int i __attribute__((no_split_stack)); // expected-error {{'no_split_stack' attribute only applies to functions}}

void f1() __attribute__((no_split_stack));
void f2() __attribute__((no_split_stack(1))); // expected-error {{'no_split_stack' attribute takes no arguments}}

template <typename T>
void tf1() __attribute__((no_split_stack));

int f3(int __attribute__((no_split_stack)), int); // expected-error{{'no_split_stack' attribute only applies to functions}}

struct A {
  int f __attribute__((no_split_stack));  // expected-error{{'no_split_stack' attribute only applies to functions}}
  void mf1() __attribute__((no_split_stack));
  static void mf2() __attribute__((no_split_stack));
};

int ci [[gnu::no_split_stack]]; // expected-error {{'no_split_stack' attribute only applies to functions}}

[[gnu::no_split_stack]] void cf1();
[[gnu::no_split_stack(1)]] void cf2(); // expected-error {{'no_split_stack' attribute takes no arguments}}

template <typename T>
[[gnu::no_split_stack]]
void ctf1();

int cf3(int c[[gnu::no_split_stack]], int); // expected-error{{'no_split_stack' attribute only applies to functions}}

struct CA {
  int f [[gnu::no_split_stack]];  // expected-error{{'no_split_stack' attribute only applies to functions}}
  [[gnu::no_split_stack]] void mf1();
  [[gnu::no_split_stack]] static void mf2();
};
