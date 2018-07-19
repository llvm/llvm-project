// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -fsyntax-only -Wignored-qualifiers -Wno-error=return-type -verify -fblocks -Wno-unreachable-code -Wno-unused-value

// clang emits the following warning by default.
// With GCC, -pedantic, -Wreturn-type or -Wall are required to produce the 
// following warning.
int t14() {
  return; // expected-warning {{non-void function 't14' should return a value}}
}

void t15() {
  return 1; // expected-warning {{void function 't15' should not return a value}}
}

int unknown();

void test0() {
}

int test1() {
} // expected-warning {{control reaches end of non-void function}}

int test2() {
  a: goto a;
}

int test3() {
  goto a;
  a: ;
} // expected-warning {{control reaches end of non-void function}}


void halt() {
  a: goto a;
}

void halt2() __attribute__((noreturn));

int test4() {
  halt2();
}

int test5() {
  halt2(), (void)1;
}

int test6() {
  1, halt2();
}

int j;
int unknown_nohalt() {
  return j;
}

int test7() {
  unknown();
} // expected-warning {{control reaches end of non-void function}}

int test8() {
  (void)(1 + unknown());
} // expected-warning {{control reaches end of non-void function}}

int halt3() __attribute__((noreturn));

int test9() {
  (void)(halt3() + unknown());
}

int test10() {
  (void)(unknown() || halt3());
} // expected-warning {{control may reach end of non-void function}}

int test11() {
  (void)(unknown() && halt3());
} // expected-warning {{control may reach end of non-void function}}

int test12() {
  (void)(halt3() || unknown());
}

int test13() {
  (void)(halt3() && unknown());
}

int test14() {
  (void)(1 || unknown());
} // expected-warning {{control reaches end of non-void function}}

int test15() {
  (void)(0 || unknown());
} // expected-warning {{control reaches end of non-void function}}

int test16() {
  (void)(0 && unknown());
} // expected-warning {{control reaches end of non-void function}}

int test17() {
  (void)(1 && unknown());
} // expected-warning {{control reaches end of non-void function}}

int test18() {
  (void)(unknown_nohalt() && halt3());
} // expected-warning {{control may reach end of non-void function}}

int test19() {
  (void)(unknown_nohalt() && unknown());
} // expected-warning {{control reaches end of non-void function}}

int test20() {
  int i;
  if (i)
    return 0;
  else if (0)
    return 2;
} // expected-warning {{control may reach end of non-void function}}

int test21() {
  int i;
  if (i)
    return 0;
  else if (1)
    return 2;
}

int test22() {
  int i;
  switch (i) default: ;
} // expected-warning {{control reaches end of non-void function}}

int test23() {
  int i;
  switch (i) {
  case 0:
    return 0;
  case 2:
    return 2;
  }
} // expected-warning {{control may reach end of non-void function}}

int test24() {
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

int test25() {
  1 ? halt3() : unknown();
}

int test26() {
  0 ? halt3() : unknown();
} // expected-warning {{control reaches end of non-void function}}

int j;
void (*fptr)() __attribute__((noreturn));
int test27() {
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
int test29() {
  exit(1);
}

// Include these declarations here explicitly so we don't depend on system headers.
typedef struct __jmp_buf_tag{} jmp_buf[1];

extern void longjmp (struct __jmp_buf_tag __env[1], int __val) __attribute__ ((noreturn));
extern void _longjmp (struct __jmp_buf_tag __env[1], int __val) __attribute__ ((noreturn));

jmp_buf test30_j;

int test30() {
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

void test32() {
  ^ (void) { while (1) { } }();
  ^ (void) { if (j) while (1) { } }();
  while (1) { }
}

void test33() {
  if (j) while (1) { }
}

// Test that 'static inline' functions are only analyzed for CFG-based warnings
// when they are used.
static inline int si_has_missing_return() {} // expected-warning{{control reaches end of non-void function}}
static inline int si_has_missing_return_2() {}; // expected-warning{{control reaches end of non-void function}}
static inline int si_forward();
static inline int si_has_missing_return_3(int x) {
  if (x)
   return si_has_missing_return_3(x+1);
} // expected-warning{{control may reach end of non-void function}}

int test_static_inline(int x) {
  si_forward();
  return x ? si_has_missing_return_2() : si_has_missing_return_3(x);
}
static inline int si_forward() {} // expected-warning{{control reaches end of non-void function}}

// Test warnings on ignored qualifiers on return types.
const int ignored_c_quals(); // expected-warning{{'const' type qualifier on return type has no effect}}
const volatile int ignored_cv_quals(); // expected-warning{{'const volatile' type qualifiers on return type have no effect}}
char* const volatile restrict ignored_cvr_quals(); // expected-warning{{'const volatile restrict' type qualifiers on return type have no effect}}

typedef const int CI;
CI ignored_quals_typedef();

const CI ignored_quals_typedef_2(); // expected-warning{{'const' type qualifier}}

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
int test35() {
lbl:
  if (1)
    goto lbl;
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
} // expected-warning {{control may reach end of non-void function}}

// sizeof(long) test.
int sizeof_long() {
  if (sizeof(long) == 4)
    return 1;
  if (sizeof(long) == 8)
    return 2;
} // no-warning
