// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -S -verify -o -

struct S {
  __attribute__((always_inline, target("avx512f")))
  void foo(){}
  __attribute__((always_inline, target("avx512f")))
  operator int(){ return 0; }
  __attribute__((always_inline, target("avx512f")))
  void operator()(){ }

};

void usage(S & s) {
  s.foo(); // expected-error {{'foo' requires target feature 'avx512f'}}
  (void)(int)s; // expected-error {{'operator int' requires target feature 'avx512f'}}
  s(); // expected-error {{'operator()' requires target feature 'avx512f'}}
}
