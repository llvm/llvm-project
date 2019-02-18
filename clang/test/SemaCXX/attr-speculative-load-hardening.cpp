// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

int i __attribute__((speculative_load_hardening)); // expected-error {{'speculative_load_hardening' attribute only applies to functions}}

void f1() __attribute__((speculative_load_hardening));
void f2() __attribute__((speculative_load_hardening(1))); // expected-error {{'speculative_load_hardening' attribute takes no arguments}}

template <typename T>
void tf1() __attribute__((speculative_load_hardening));

int f3(int __attribute__((speculative_load_hardening)), int); // expected-error {{'speculative_load_hardening' attribute only applies to functions}}

struct A {
  int f __attribute__((speculative_load_hardening));  // expected-error {{'speculative_load_hardening' attribute only applies to functions}}
  void mf1() __attribute__((speculative_load_hardening));
  static void mf2() __attribute__((speculative_load_hardening));
};

int ci [[clang::speculative_load_hardening]]; // expected-error {{'speculative_load_hardening' attribute only applies to functions}}

[[clang::speculative_load_hardening]] void cf1();
[[clang::speculative_load_hardening(1)]] void cf2(); // expected-error {{'speculative_load_hardening' attribute takes no arguments}}

template <typename T>
[[clang::speculative_load_hardening]]
void ctf1();

int cf3(int c[[clang::speculative_load_hardening]], int); // expected-error {{'speculative_load_hardening' attribute only applies to functions}}

struct CA {
  int f [[clang::speculative_load_hardening]];  // expected-error {{'speculative_load_hardening' attribute only applies to functions}}
  [[clang::speculative_load_hardening]] void mf1();
  [[clang::speculative_load_hardening]] static void mf2();
};
