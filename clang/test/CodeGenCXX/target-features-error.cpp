// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -S -verify -o - -DTEST1
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -S -verify -o - -DTEST2
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -S -verify -o - -DTEST3

struct S {
  __attribute__((always_inline, target("avx512f")))
  void foo(){}
  __attribute__((always_inline, target("avx512f")))
  operator int(){ return 0; }
  __attribute__((always_inline, target("avx512f")))
  void operator()(){ }

};
__attribute__((always_inline, target("avx512f")))
void free_func(){}


#ifdef TEST1
void usage(S & s) {
  s.foo(); // expected-error {{'foo' requires target feature 'avx512f'}}
  (void)(int)s; // expected-error {{'operator int' requires target feature 'avx512f'}}
  s(); // expected-error {{'operator()' requires target feature 'avx512f'}}
  free_func(); // expected-error{{'free_func' requires target feature 'avx512f'}}

}
#endif

#ifdef TEST2
__attribute__((target("avx512f")))
void usage(S & s) {
  s.foo();
  (void)(int)s;
  s();

  [&s] {
    s.foo();       // expected-error {{'foo' requires target feature 'avx512f'}}
    (void)(int) s; // expected-error {{'operator int' requires target feature 'avx512f'}}
    s();           // expected-error {{'operator()' requires target feature 'avx512f'}}
    free_func();   // expected-error{{'free_func' requires target feature 'avx512f'}}
  }();
}
#endif

#ifdef TEST3
void usage(S & s) {

  [&s] () __attribute__((target("avx512f"))) {
    s.foo();
    (void)(int) s;
    s();
    free_func();
  }();

  [&s] {
    s.foo();       // expected-error {{'foo' requires target feature 'avx512f'}}
    (void)(int) s; // expected-error {{'operator int' requires target feature 'avx512f'}}
    s();           // expected-error {{'operator()' requires target feature 'avx512f'}}
    free_func();   // expected-error{{'free_func' requires target feature 'avx512f'}}
  }();
}
#endif
