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

void f4() __attribute__((no_speculative_load_hardening, speculative_load_hardening)); // expected-error {{attributes are not compatible}}
// expected-note@-1 {{conflicting attribute is here}}

void f5() __attribute__((speculative_load_hardening, no_speculative_load_hardening)); // expected-error {{attributes are not compatible}}
// expected-note@-1 {{conflicting attribute is here}}

void f6() __attribute__((no_speculative_load_hardening));

void f6() __attribute__((speculative_load_hardening)); // expected-error@-2 {{'no_speculative_load_hardening' and 'speculative_load_hardening' attributes are not compatible}}
// expected-note@-1 {{conflicting attribute is here}}

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

[[clang::speculative_load_hardening, clang::no_speculative_load_hardening]] void cf4();  // expected-error {{attributes are not compatible}}
// expected-note@-1 {{conflicting attribute is here}}

[[clang::no_speculative_load_hardening, clang::speculative_load_hardening]] void cf5();  // expected-error {{attributes are not compatible}}
// expected-note@-1 {{conflicting attribute is here}}

[[clang::speculative_load_hardening]]
void cf6();

[[clang::no_speculative_load_hardening]]
void cf6(); // expected-error@-4 {{'speculative_load_hardening' and 'no_speculative_load_hardening' attributes are not compatible}} \
// expected-note@-1 {{conflicting attribute is here}}
