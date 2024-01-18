// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=security.cert.env.InvalidPtr\
// RUN:  -analyzer-config security.cert.env.InvalidPtr:InvalidatingGetEnv=true \
// RUN:  -analyzer-output=text -verify -Wno-unused %s

#include "../Inputs/system-header-simulator.h"
char *getenv(const char *name);
char *setlocale(int category, const char *locale);
char *strerror(int errnum);

typedef struct {
  char * field;
} lconv;
lconv *localeconv(void);

typedef struct {
} tm;
char *asctime(const tm *timeptr);

int strcmp(const char*, const char*);
extern void foo(char *e);
extern char* bar(void);


void getenv_test1(void) {
  char *p;

  p = getenv("VAR");
  *p; // no-warning

  p = getenv("VAR2");
  *p; // no-warning, getenv result was assigned to the same pointer
}

void getenv_test2(void) {
  char *p, *p2;

  p = getenv("VAR");
  // expected-note@-1{{previous function call was here}}
  *p; // no-warning

  p2 = getenv("VAR2");
  // expected-note@-1{{'getenv' call may invalidate the result of the previous 'getenv'}}

  *p;
  // expected-warning@-1{{dereferencing an invalid pointer}}
  // expected-note@-2{{dereferencing an invalid pointer}}
}

void getenv_test3(void) {
  char *p, *p2, *p3;

  p = getenv("VAR");
  *p; // no-warning

  p = getenv("VAR2");
  // expected-note@-1{{previous function call was here}}
  p2 = getenv("VAR2");
  // expected-note@-1{{'getenv' call may invalidate the result of the previous 'getenv'}}

  p3 = getenv("VAR3");

  *p;
  // expected-warning@-1{{dereferencing an invalid pointer}}
  // expected-note@-2{{dereferencing an invalid pointer}}
}

void getenv_test4(void) {
  char *p, *p2, *p3;

  p = getenv("VAR");
  // expected-note@-1{{previous function call was here}}
  p2 = getenv("VAR2");
  // expected-note@-1{{'getenv' call may invalidate the result of the previous 'getenv'}}
  p3 = getenv("VAR3");

  *p;
  // expected-warning@-1{{dereferencing an invalid pointer}}
  // expected-note@-2{{dereferencing an invalid pointer}}
}

void getenv_test5(void) {
  char *p, *p2, *p3;

  p = getenv("VAR");
  p2 = getenv("VAR2");
  // expected-note@-1{{previous function call was here}}
  p3 = getenv("VAR3");
  // expected-note@-1{{'getenv' call may invalidate the result of the previous 'getenv'}}

  *p2;
  // expected-warning@-1{{dereferencing an invalid pointer}}
  // expected-note@-2{{dereferencing an invalid pointer}}
}

void getenv_test6(void) {
  char *p, *p2;
  p = getenv("VAR");
  *p; // no-warning

  p = getenv("VAR2");
  // expected-note@-1{{previous function call was here}}
  *p; // no-warning

  p2 = getenv("VAR3");
  // expected-note@-1{{previous function call was here}}
  // expected-note@-2{{'getenv' call may invalidate the result of the previous 'getenv'}}

  *p;
  // expected-warning@-1{{dereferencing an invalid pointer}}
  // expected-note@-2{{dereferencing an invalid pointer}}

  *p2; // no-warning

  p = getenv("VAR4");
  // expected-note@-1{{'getenv' call may invalidate the result of the previous 'getenv'}}

  *p; // no-warning
  *p2;
  // expected-warning@-1{{dereferencing an invalid pointer}}
  // expected-note@-2{{dereferencing an invalid pointer}}
}

void getenv_test7(void) {
  char *p, *p2;
  p = getenv("VAR");
  // expected-note@-1{{previous function call was here}}
  *p; // no-warning

  p2 = getenv("VAR2");
  // expected-note@-1{{'getenv' call may invalidate the result of the previous 'getenv'}}

  foo(p);
  // expected-warning@-1{{use of invalidated pointer 'p' in a function call}}
  // expected-note@-2{{use of invalidated pointer 'p' in a function call}}
}

void getenv_test8(void) {
  static const char *array[] = {
     0,
     0,
     "/var/tmp",
     "/usr/tmp",
     "/tmp",
     "."
  };

  if( !array[0] )
  // expected-note@-1{{Taking true branch}}
    array[0] = getenv("TEMPDIR");
    // expected-note@-1{{previous function call was here}}

  if( !array[1] )
  // expected-note@-1{{Taking true branch}}
    array[1] = getenv("TMPDIR");
  // expected-note@-1{{'getenv' call may invalidate the result of the previous 'getenv'}}

  *array[0];
  // expected-warning@-1{{dereferencing an invalid pointer}}
  // expected-note@-2{{dereferencing an invalid pointer}}
}

void getenv_test9(void) {
  char *p, *p2;
  p = getenv("something");
  p = bar();
  p2 = getenv("something");
  *p; // no-warning: p does not point to getenv anymore
}

void getenv_test10(void) {
  strcmp(getenv("VAR1"), getenv("VAR2"));
  // expected-note@-1{{'getenv' call may invalidate the result of the previous 'getenv'}}
  // expected-note@-2{{previous function call was here}}
  // expected-warning@-3{{use of invalidated pointer 'getenv("VAR1")' in a function call}}
  // expected-note@-4{{use of invalidated pointer 'getenv("VAR1")' in a function call}}
}

void dereference_pointer(char* a) {
  *a;
  // expected-warning@-1{{dereferencing an invalid pointer}}
  // expected-note@-2{{dereferencing an invalid pointer}}
}

void getenv_test11(void) {
  char *p = getenv("VAR");
  // expected-note@-1{{previous function call was here}}

  char *pp = getenv("VAR2");
  // expected-note@-1{{'getenv' call may invalidate the result of the previous 'getenv'}}

  dereference_pointer(p);
  // expected-note@-1{{Calling 'dereference_pointer'}}
}

void getenv_test12(int flag1, int flag2) {
  char *p = getenv("VAR");
  // expected-note@-1{{previous function call was here}}

  if (flag1) {
    // expected-note@-1{{Assuming 'flag1' is not equal to 0}}
    // expected-note@-2{{Taking true branch}}
    char *pp = getenv("VAR2");
    // expected-note@-1{{'getenv' call may invalidate the result of the previous 'getenv'}}
  }

  if (flag2) {
    // expected-note@-1{{Assuming 'flag2' is not equal to 0}}
    // expected-note@-2{{Taking true branch}}
    *p;
    // expected-warning@-1{{dereferencing an invalid pointer}}
    // expected-note@-2{{dereferencing an invalid pointer}}
  }
}

void setlocale_test1(void) {
  char *p, *p2;
  p = setlocale(0, "VAR");
  *p; // no-warning

  p = setlocale(0, "VAR2");
  // expected-note@-1{{previous function call was here}}
  *p; // no-warning

  p2 = setlocale(0, "VAR3");
  // expected-note@-1{{'setlocale' call may invalidate the result of the previous 'setlocale'}}

  *p;
  // expected-warning@-1{{dereferencing an invalid pointer}}
  // expected-note@-2{{dereferencing an invalid pointer}}
}

void setlocale_test2(int flag) {
  char *p, *p2;
  p = setlocale(0, "VAR");
  *p; // no-warning

  p = setlocale(0, "VAR2");
  // expected-note@-1{{previous function call was here}}
  *p; // no-warning

  if (flag) {
    // expected-note@-1{{Assuming 'flag' is not equal to 0}}
    // expected-note@-2{{Taking true branch}}
    p2 = setlocale(0, "VAR3");
    // expected-note@-1{{'setlocale' call may invalidate the result of the previous 'setlocale'}}
  }

  *p;
  // expected-warning@-1{{dereferencing an invalid pointer}}
  // expected-note@-2{{dereferencing an invalid pointer}}
}

void strerror_test1(void) {
  char *p, *p2;

  p = strerror(0);
  *p; // no-warning

  p = strerror(1);
  // expected-note@-1{{previous function call was here}}
  *p; // no-warning

  p2 = strerror(2);
  // expected-note@-1{{'strerror' call may invalidate the result of the previous 'strerror'}}

  *p;
  // expected-warning@-1{{dereferencing an invalid pointer}}
  // expected-note@-2{{dereferencing an invalid pointer}}
}

void strerror_test2(int errno) {
  char *p, *p2;

  p = strerror(0);
  *p; // no-warning

  p = strerror(1);
  // expected-note@-1{{previous function call was here}}
  *p; // no-warning

  if (0 == 1) {
    // expected-note@-1{{0 is not equal to 1}}
    // expected-note@-2{{Taking false branch}}
    p2 = strerror(2);
  }

  *p; // no-warning

  if (errno) {
    // expected-note@-1{{Assuming 'errno' is not equal to 0}}
    // expected-note@-2{{Taking true branch}}
    p2 = strerror(errno);
    // expected-note@-1{{'strerror' call may invalidate the result of the previous 'strerror'}}
  }

  *p;
  // expected-warning@-1{{dereferencing an invalid pointer}}
  // expected-note@-2{{dereferencing an invalid pointer}}
}

void asctime_test(void) {
  const tm *t;
  const tm *tt;

  char* p = asctime(t);
  // expected-note@-1{{previous function call was here}}
  char* pp = asctime(tt);
  // expected-note@-1{{'asctime' call may invalidate the result of the previous 'asctime'}}

  *p;
  // expected-warning@-1{{dereferencing an invalid pointer}}
  // expected-note@-2{{dereferencing an invalid pointer}}
}

void localeconv_test1(void) {
  lconv *lc1 = localeconv();
  // expected-note@-1{{previous function call was here}}
  lconv *lc2 = localeconv();
  // expected-note@-1{{'localeconv' call may invalidate the result of the previous 'localeconv'}}

  *lc1;
  // expected-warning@-1{{dereferencing an invalid pointer}}
  // expected-note@-2{{dereferencing an invalid pointer}}
}

void localeconv_test2(void) {
  // TODO: false negative
  lconv *lc1 = localeconv();
  lconv *lc2 = localeconv();
  lc1->field;
}
