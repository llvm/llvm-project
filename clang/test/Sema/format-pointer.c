// RUN: %clang_cc1 -Wformat %s -verify
// RUN: %clang_cc1 -Wformat -std=c23 %s -verify
// RUN: %clang_cc1 -xc++ -Wformat %s -verify
// RUN: %clang_cc1 -xobjective-c -Wformat -fblocks %s -verify
// RUN: %clang_cc1 -xobjective-c++ -Wformat -fblocks %s -verify
// RUN: %clang_cc1 -std=c23 -Wformat %s -pedantic -verify=expected,pedantic
// RUN: %clang_cc1 -xc++ -Wformat %s -pedantic -verify=expected,pedantic
// RUN: %clang_cc1 -xobjective-c -Wformat -fblocks -pedantic %s -verify=expected,pedantic

__attribute__((__format__(__printf__, 1, 2)))
int printf(const char *, ...);
__attribute__((__format__(__scanf__, 1, 2)))
int scanf(const char *, ...);

void f(void *vp, const void *cvp, char *cp, signed char *scp, int *ip) {
  int arr[2];

  printf("%p", cp);
  printf("%p", cvp);
  printf("%p", vp);
  printf("%p", scp);
  printf("%p", ip); // pedantic-warning {{format specifies type 'void *' but the argument has type 'int *'}}
  printf("%p", arr); // pedantic-warning {{format specifies type 'void *' but the argument has type 'int *'}}

  scanf("%p", &vp);
  scanf("%p", &cvp);
  scanf("%p", (void *volatile*)&vp);
  scanf("%p", (const void *volatile*)&cvp);
  scanf("%p", &cp); // pedantic-warning {{format specifies type 'void **' but the argument has type 'char **'}}
  scanf("%p", &ip); // pedantic-warning {{format specifies type 'void **' but the argument has type 'int **'}}
  scanf("%p", &arr); // expected-warning {{format specifies type 'void **' but the argument has type 'int (*)[2]'}}

#if !__is_identifier(nullptr)
  typedef __typeof__(nullptr) nullptr_t;
  nullptr_t np = nullptr;
  nullptr_t *npp = &np;

  printf("%p", np);
  scanf("%p", &np); // expected-warning {{format specifies type 'void **' but the argument has type 'nullptr_t *'}}
  scanf("%p", &npp); // pedantic-warning {{format specifies type 'void **' but the argument has type 'nullptr_t **'}}
#endif

#ifdef __OBJC__
  id i = 0;
  void (^b)(void) = ^{};

  printf("%p", i); // pedantic-warning {{format specifies type 'void *' but the argument has type 'id'}}
  printf("%p", b); // pedantic-warning {{format specifies type 'void *' but the argument has type 'void (^)(void)'}}
  scanf("%p", &i); // pedantic-warning {{format specifies type 'void **' but the argument has type 'id *'}}
  scanf("%p", &b); // pedantic-warning {{format specifies type 'void **' but the argument has type 'void (^*)(void)'}}
#endif

}
