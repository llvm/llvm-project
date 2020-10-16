// RUN: %clang_cc1 -triple i386-mingw32 -fms-extensions -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify %s

int nonconst(void);
int isconst(void) __attribute__((const));
int ispure(int) __attribute__((pure));

int foo(int *a, int i) {
#ifdef _MSC_VER
  __assume(i != 4);
  __assume(++i > 2); //expected-warning {{the argument to '__assume' has side effects that will be discarded}}
  __assume(nonconst() > 2); //expected-warning {{the argument to '__assume' has side effects that will be discarded}}
  __assume(isconst() > 2);
  __assume(ispure(i) > 2);
  __assume(ispure(++i) > 2); //expected-warning {{the argument to '__assume' has side effects that will be discarded}}

  int test = sizeof(struct{char qq[(__assume(i != 5), 7)];});
#else
  __builtin_assume(i != 4);
  __builtin_assume(++i > 2); //expected-warning {{the argument to '__builtin_assume' has side effects that will be discarded}}
  __builtin_assume(nonconst() > 2); //expected-warning {{the argument to '__builtin_assume' has side effects that will be discarded}}
  __builtin_assume(isconst() > 2);
  __builtin_assume(ispure(i) > 2);
  __builtin_assume(ispure(++i) > 2); //expected-warning {{the argument to '__builtin_assume' has side effects that will be discarded}}
  
  int test = sizeof(struct{char qq[(__builtin_assume(i != 5), 7)];}); // expected-warning {{variable length array}}
#endif
  return a[i];
}

