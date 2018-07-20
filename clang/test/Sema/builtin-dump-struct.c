// RUN: %clang_cc1 -triple i386-unknown-unknown -fsyntax-only -fno-spell-checking -verify %s

void invalid_uses() {
  struct A {
  };
  struct A a;
  void *b;
  int (*goodfunc)(const char *, ...);
  int (*badfunc1)(const char *);
  int (*badfunc2)(int, ...);
  void (*badfunc3)(const char *, ...);
  int (*badfunc4)(char *, ...);
  int (*badfunc5)(void);

  __builtin_dump_struct();             // expected-error {{too few arguments to function call, expected 2, have 0}}
  __builtin_dump_struct(1);            // expected-error {{too few arguments to function call, expected 2, have 1}}
  __builtin_dump_struct(1, 2);         // expected-error {{passing 'int' to parameter of incompatible type structure pointer: type mismatch at 1st parameter ('int' vs structure pointer)}}
  __builtin_dump_struct(&a, 2);        // expected-error {{passing 'int' to parameter of incompatible type 'int (*)(const char *, ...)': type mismatch at 2nd parameter ('int' vs 'int (*)(const char *, ...)')}}
  __builtin_dump_struct(b, goodfunc); // expected-error {{passing 'void *' to parameter of incompatible type structure pointer: type mismatch at 1st parameter ('void *' vs structure pointer)}}
  __builtin_dump_struct(&a, badfunc1); // expected-error {{passing 'int (*)(const char *)' to parameter of incompatible type 'int (*)(const char *, ...)': type mismatch at 2nd parameter ('int (*)(const char *)' vs 'int (*)(const char *, ...)')}}
  __builtin_dump_struct(&a, badfunc2); // expected-error {{passing 'int (*)(int, ...)' to parameter of incompatible type 'int (*)(const char *, ...)': type mismatch at 2nd parameter ('int (*)(int, ...)' vs 'int (*)(const char *, ...)')}}
  __builtin_dump_struct(&a, badfunc3); // expected-error {{passing 'void (*)(const char *, ...)' to parameter of incompatible type 'int (*)(const char *, ...)': type mismatch at 2nd parameter ('void (*)(const char *, ...)' vs 'int (*)(const char *, ...)')}}
  __builtin_dump_struct(&a, badfunc4); // expected-error {{passing 'int (*)(char *, ...)' to parameter of incompatible type 'int (*)(const char *, ...)': type mismatch at 2nd parameter ('int (*)(char *, ...)' vs 'int (*)(const char *, ...)')}}
  __builtin_dump_struct(&a, badfunc5); // expected-error {{passing 'int (*)(void)' to parameter of incompatible type 'int (*)(const char *, ...)': type mismatch at 2nd parameter ('int (*)(void)' vs 'int (*)(const char *, ...)')}}
  __builtin_dump_struct(a, goodfunc);  // expected-error {{passing 'struct A' to parameter of incompatible type structure pointer: type mismatch at 1st parameter ('struct A' vs structure pointer)}}
}

void valid_uses() {
  struct A {
  };
  union B {
  };

  int (*goodfunc)(const char *, ...);
  int (*goodfunc2)();
  struct A a;
  union B b;

  __builtin_dump_struct(&a, goodfunc);
  __builtin_dump_struct(&b, goodfunc);
  __builtin_dump_struct(&a, goodfunc2);
}
