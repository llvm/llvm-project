// RUN: %clang_cc1 %s -fsyntax-only -verify
// rdar://10095762

typedef void (*Fn_noret)(void) __attribute__((noreturn));
typedef void (*Fn_ret)(void);

typedef void (*Fn_noret_noproto)() __attribute__((noreturn));
typedef void (*Fn_ret_noproto)();

void foo(void);
void foo_noret(void)  __attribute__((noreturn));

void foo_noproto();
void foo_noret_noproto()  __attribute__((noreturn));

void test() {
  Fn_noret fn2 = &foo; // expected-warning {{incompatible function pointer types initializing 'Fn_noret'}}
  Fn_noret fn3 = &foo_noret; 
  Fn_ret fn4 = &foo_noret; 
  Fn_ret fn5 = &foo;

  Fn_noret_noproto fn6 = &foo_noproto; // expected-warning {{incompatible function pointer types initializing 'Fn_noret_noproto'}}
  Fn_noret_noproto fn7 = &foo_noret_noproto; 
  Fn_ret_noproto fn8 = &foo_noret_noproto; 
  Fn_ret_noproto fn9 = &foo_noproto;
}

