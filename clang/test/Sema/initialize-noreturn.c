// RUN: %clang_cc1 %s -fsyntax-only -verify
// rdar://10095762

typedef void (*Fn_noret)(void) __attribute__((noreturn));
typedef void (*Fn_ret)(void);

void foo(void);
void foo_noret(void)  __attribute__((noreturn));

void test() {
  Fn_noret fn2 = &foo; // expected-warning {{incompatible function pointer types initializing 'Fn_noret'}}
  Fn_noret fn3 = &foo_noret; 
  Fn_ret fn4 = &foo_noret; 
  Fn_ret fn5 = &foo;
}

