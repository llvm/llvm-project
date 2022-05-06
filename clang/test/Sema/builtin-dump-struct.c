// RUN: %clang_cc1 -triple i386-unknown-unknown -fsyntax-only -fno-spell-checking -Wno-strict-prototypes -verify %s -fblocks

void invalid_uses(void) {
  struct A {
  };
  struct A a;
  void *b;
  int (*goodfunc)(const char *, ...);
  int (*badfunc1)(const char *);
  int (*badfunc2)(int, ...);
  int (*badfunc3)(void);

  __builtin_dump_struct();             // expected-error {{too few arguments to function call, expected 2, have 0}}
  __builtin_dump_struct(1);            // expected-error {{too few arguments to function call, expected 2, have 1}}
  __builtin_dump_struct(1, 2);         // expected-error {{expected pointer to struct as 1st argument to '__builtin_dump_struct', found 'int'}}
  __builtin_dump_struct(&a, 2);        // expected-error {{expected a callable expression as 2nd argument to '__builtin_dump_struct', found 'int'}}
  __builtin_dump_struct(b, goodfunc); // expected-error {{expected pointer to struct as 1st argument to '__builtin_dump_struct', found 'void *'}}
  __builtin_dump_struct(&a, badfunc1); // expected-error {{too many arguments to function call, expected 1, have 2}} expected-note  {{in call to printing function with arguments '("%s", "struct A")'}}
  __builtin_dump_struct(&a, badfunc2); // expected-warning-re 1+{{incompatible pointer to integer conversion passing 'char[{{.*}}]' to parameter of type 'int'}}
                                       // expected-note@-1 1+{{in call to printing function with arguments '("}}
  __builtin_dump_struct(&a, badfunc3); // expected-error {{too many arguments to function call, expected 0, have 2}} expected-note {{in call to printing function with arguments '("%s", "struct A")'}}
  __builtin_dump_struct(a, goodfunc);  // expected-error {{expected pointer to struct as 1st argument to '__builtin_dump_struct', found 'struct A'}}
}

int goodglobalfunc(const char*, ...);

void valid_uses(void) {
  struct A {
  };
  union B {
  };

  int (*goodfunc)(const char *, ...);
  int (*goodfunc2)();
  void (*goodfunc3)(const char *, ...);
  int (*goodfunc4)(char *, ...);
  int (^goodblock)(const char*, ...);
  struct A a;
  union B b;

  __builtin_dump_struct(&a, goodglobalfunc);
  __builtin_dump_struct(&a, &goodglobalfunc);
  __builtin_dump_struct(&a, goodfunc);
  __builtin_dump_struct(&b, goodfunc);
  __builtin_dump_struct(&a, goodfunc2);
  __builtin_dump_struct(&a, goodfunc3);
  __builtin_dump_struct(&a, goodfunc4);
  __builtin_dump_struct(&a, goodblock);
}
