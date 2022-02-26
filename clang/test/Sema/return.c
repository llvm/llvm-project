// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -fsyntax-only -Wignored-qualifiers -Wno-error=return-type -verify -fblocks -Wno-unreachable-code -Wno-unused-value -Wno-strict-prototypes

// clang emits the following warning by default.
// With GCC, -pedantic, -Wreturn-type or -Wall are required to produce the 
// following warning.
int t14(void) {
  return; // expected-warning {{non-void function 't14' should return a value}}
}

void t15(void) {
  return 1; // expected-warning {{void function 't15' should not return a value}}
}

int unknown(void);

void test0(void) {
}

int test1(void) {
} // expected-warning {{non-void function does not return a value}}

int test2(void) {
  a: goto a;
}

int test3(void) {
  goto a;
  a: ;
} // expected-warning {{non-void function does not return a value}}


void halt(void) {
  a: goto a;
}

void halt2(void) __attribute__((noreturn));

int test4(void) {
  halt2();
}

int test5(void) {
  halt2(), (void)1;
}

int test6(void) {
  1, halt2();
}

int j;
int unknown_nohalt(void) {
  return j;
}

int test7(void) {
  unknown();
} // expected-warning {{non-void function does not return a value}}

int test8(void) {
  (void)(1 + unknown());
} // expected-warning {{non-void function does not return a value}}

int halt3(void) __attribute__((noreturn));

int test9(void) {
  (void)(halt3() + unknown());
}

int test10(void) {
  (void)(unknown() || halt3());
} // expected-warning {{non-void function does not return a value in all control paths}}

int test11(void) {
  (void)(unknown() && halt3());
} // expected-warning {{non-void function does not return a value in all control paths}}

int test12(void) {
  (void)(halt3() || unknown());
}

int test13(void) {
  (void)(halt3() && unknown());
}

int test14(void) {
  (void)(1 || unknown());
} // expected-warning {{non-void function does not return a value}}

int test15(void) {
  (void)(0 || unknown());
} // expected-warning {{non-void function does not return a value}}

int test16(void) {
  (void)(0 && unknown());
} // expected-warning {{non-void function does not return a value}}

int test17(void) {
  (void)(1 && unknown());
} // expected-warning {{non-void function does not return a value}}

int test18(void) {
  (void)(unknown_nohalt() && halt3());
} // expected-warning {{non-void function does not return a value in all control paths}}

int test19(void) {
  (void)(unknown_nohalt() && unknown());
} // expected-warning {{non-void function does not return a value}}

int test20(void) {
  int i;
  if (i)
    return 0;
  else if (0)
    return 2;
} // expected-warning {{non-void function does not return a value in all control paths}}

int test21(void) {
  int i;
  if (i)
    return 0;
  else if (1)
    return 2;
}

int test22(void) {
  int i;
  switch (i) default: ;
} // expected-warning {{non-void function does not return a value}}

int test23(void) {
  int i;
  switch (i) {
  case 0:
    return 0;
  case 2:
    return 2;
  }
} // expected-warning {{non-void function does not return a value in all control paths}}

int test24(void) {
  int i;
  switch (i) {
    case 0:
    return 0;
  case 2:
    return 2;
  default:
    return -1;
  }
}

int test25(void) {
  1 ? halt3() : unknown();
}

int test26(void) {
  0 ? halt3() : unknown();
} // expected-warning {{non-void function does not return a value}}

int j;
void (*fptr)(void) __attribute__((noreturn));
int test27(void) {
  switch (j) {
  case 1:
    do { } while (1);
    break;
  case 2:
    for (;;) ;
    break;
  case 3:
    for (;1;) ;
    for (;0;) {
      goto done;
    }
    return 1;
  case 4:    
    while (0) { goto done; }
    return 1;
  case 5:
    while (1) { return 1; }
    break;
  case 6:
    fptr();
    break;
  default:
    return 1;
  }
  done: ;
}

// PR4624
void test28() __attribute__((noreturn));
void test28(x) { while (1) { } }

void exit(int);
int test29(void) {
  exit(1);
}

// Include these declarations here explicitly so we don't depend on system headers.
typedef struct __jmp_buf_tag{} jmp_buf[1];

extern void longjmp (struct __jmp_buf_tag __env[1], int __val) __attribute__ ((noreturn));
extern void _longjmp (struct __jmp_buf_tag __env[1], int __val) __attribute__ ((noreturn));

jmp_buf test30_j;

int test30(void) {
  if (j)
    longjmp(test30_j, 1);
  else
#if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
    longjmp(test30_j, 2);
#else
    _longjmp(test30_j, 1);
#endif
}

typedef void test31_t(int status);
void test31(test31_t *callback __attribute__((noreturn)));

void test32(void) {
  ^ (void) { while (1) { } }();
  ^ (void) { if (j) while (1) { } }();
  while (1) { }
}

void test33(void) {
  if (j) while (1) { }
}

// Test that 'static inline' functions are only analyzed for CFG-based warnings
// when they are used.
static inline int si_has_missing_return(void) {} // expected-warning{{non-void function does not return a value}}
static inline int si_has_missing_return_2(void) {}; // expected-warning{{non-void function does not return a value}}
static inline int si_forward(void);
static inline int si_has_missing_return_3(int x) {
  if (x)
   return si_has_missing_return_3(x+1);
} // expected-warning{{non-void function does not return a value in all control paths}}

int test_static_inline(int x) {
  si_forward();
  return x ? si_has_missing_return_2() : si_has_missing_return_3(x);
}
static inline int si_forward(void) {} // expected-warning{{non-void function does not return a value}}

// Test warnings on ignored qualifiers on return types.
const int ignored_c_quals(void); // expected-warning{{'const' type qualifier on return type has no effect}}
const volatile int ignored_cv_quals(void); // expected-warning{{'const volatile' type qualifiers on return type have no effect}}
char* const volatile restrict ignored_cvr_quals(void); // expected-warning{{'const volatile restrict' type qualifiers on return type have no effect}}

typedef const int CI;
CI ignored_quals_typedef(void);

const CI ignored_quals_typedef_2(void); // expected-warning{{'const' type qualifier}}

// Test that for switch(enum) that if the switch statement covers all the cases
// that we don't consider that for -Wreturn-type.
enum Cases { C1, C2, C3, C4 };
int test_enum_cases(enum Cases C) {
  switch (C) {
  case C1: return 1;
  case C2: return 2;
  case C4: return 3;
  case C3: return 4;
  }
} // no-warning

// PR12318 - Don't give a may reach end of non-void function warning.
int test34(int x) {
  if (x == 1) {
    return 3;
  } else if ( x == 2 || 1) {
    return 5;
  }
}

// PR18999
int test35(void) {
lbl:
  if (1)
    goto lbl;
}

int test36a(int b) {
  if (b)
    return 43;
  __builtin_unreachable();
}

int test36b(int b) {
  if (b)
    return 43;
  __builtin_assume(0);
}

// PR19074.
void abort(void) __attribute__((noreturn));
#define av_assert0(cond) do {\
    if (!(cond)) {\
      abort();\
    }\
  } while (0)

int PR19074(int x) {
  switch(x) {
  case 0:
    return 0;
  default:
    av_assert0(0);
  } // no-warning
}

int PR19074_positive(int x) {
  switch(x) {
  case 0:
    return 0;
  default:
    break;
  }
} // expected-warning {{non-void function does not return a value in all control paths}}

// sizeof(long) test.
int sizeof_long(void) {
  if (sizeof(long) == 4)
    return 1;
  if (sizeof(long) == 8)
    return 2;
} // no-warning

int return_statement_expression(void) {
  if (unknown())
    return ({
      while (0)
        ;
      0;
    });
  else
    return 0;
} // no-warning (used to be "non-void function does not return a value in all control paths")
