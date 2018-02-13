// RUN: %clang_cc1 %s -fsyntax-only -verify -fblocks -Wunreachable-code-aggressive -Wno-unused-value -Wno-covered-switch-default -I %S/Inputs
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wunreachable-code-aggressive -Wno-unused-value -Wno-covered-switch-default -fdiagnostics-parseable-fixits -I %S/Inputs %s 2>&1 | FileCheck %s

#include "warn-unreachable.h"

int halt() __attribute__((noreturn));
int live();
int dead();

void test1() {
  goto c;
  d:
  goto e;       // expected-warning {{will never be executed}}
  c: ;
  int i;
  return;
  goto b;        // expected-warning {{will never be executed}}
  goto a;        // expected-warning {{will never be executed}}
  b:
  i = 1;
  a:
  i = 2;
  goto f;
  e:
  goto d;
  f: ;
}

void test2() {
  int i;
  switch (live()) {
  case 1:
    halt(),
      dead();   // expected-warning {{will never be executed}}

  case 2:
    live(), halt(),
      dead();   // expected-warning {{will never be executed}}

  case 3:
  live()
    +           // expected-warning {{will never be executed}}
    halt();
  dead();

  case 4:
  a4:
    live(),
      halt();
    goto a4;    // expected-warning {{will never be executed}}

  case 5:
    goto a5;
  c5:
    dead();     // expected-warning {{will never be executed}}
    goto b5;
  a5:
    live(),
      halt();
  b5:
    goto c5;

  case 6:
    if (live())
      goto e6;
    live(),
      halt();
  d6:
    dead();     // expected-warning {{will never be executed}}
    goto b6;
  c6:
    dead();
    goto b6;
  e6:
    live(),
      halt();
  b6:
    goto c6;
  case 7:
    halt()
      +
      dead();   // expected-warning {{will never be executed}}
    -           // expected-warning {{will never be executed}}
      halt();
  case 8:
    i
      +=        // expected-warning {{will never be executed}}
      halt();
  case 9:
    halt()
      ?         // expected-warning {{will never be executed}}
      dead() : dead();
  case 10:
    (           // expected-warning {{will never be executed}}
      float)halt();
  case 11: {
    int a[5];
    live(),
      a[halt()
        ];      // expected-warning {{will never be executed}}
  }
  }
}

enum Cases { C1, C2, C3 };
int test_enum_cases(enum Cases C) {
  switch (C) {
    case C1:
    case C2:
    case C3:
      return 1;
    default: {
      int i = 0; // no-warning
      ++i;
      return i;
    }
  }  
}

// Handle unreachable code triggered by macro expansions.
void __myassert_rtn(const char *, const char *, int, const char *) __attribute__((__noreturn__));

#define myassert(e) \
    (__builtin_expect(!(e), 0) ? __myassert_rtn(__func__, __FILE__, __LINE__, #e) : (void)0)

void test_assert() {
  myassert(0 && "unreachable");
  return; // no-warning
}

// Test case for PR 9774.  Tests that dead code in macros aren't warned about.
#define MY_MAX(a,b)     ((a) >= (b) ? (a) : (b))
void PR9774(int *s) {
    for (int i = 0; i < MY_MAX(2, 3); i++) // no-warning
        s[i] = 0;
}

// Test case for <rdar://problem/11005770>.  We should treat code guarded
// by 'x & 0' and 'x * 0' as unreachable.
int calledFun();
void test_mul_and_zero(int x) {
  if (x & 0) calledFun(); // expected-warning {{will never be executed}}
  if (0 & x) calledFun(); // expected-warning {{will never be executed}}
  if (x * 0) calledFun(); // expected-warning {{will never be executed}}
  if (0 * x) calledFun(); // expected-warning {{will never be executed}}
}

void raze() __attribute__((noreturn));
void warn_here();

int test_break_preceded_by_noreturn(int i) {
  switch (i) {
    case 1:
      raze();
      break; // expected-warning {{'break' will never be executed}}
    case 2:
      raze();
      break; // expected-warning {{'break' will never be executed}}
      warn_here(); // expected-warning {{will never be executed}}
    case 3:
      return 1;
      break; // expected-warning {{will never be executed}}
    default:
      break;
      break; // expected-warning {{will never be executed}}
  }
  return i;
}

// Don't warn about unreachable 'default' cases, as that is covered
// by -Wcovered-switch-default.
typedef enum { Value1 = 1 } MyEnum;
void unreachable_default(MyEnum e) {
  switch (e) {
  case Value1:
    calledFun();
    break;
  case 2: // expected-warning {{case value not in enumerated type 'MyEnum'}}
    calledFun();
    break;
  default:
    calledFun(); // no-warning
    break;
  }
}
void unreachable_in_default(MyEnum e) {
  switch (e) {
  default:
    raze();
    calledFun(); // expected-warning {{will never be executed}}
    break;
  }
}

// Don't warn about trivial dead returns.
int trivial_dead_return() {
  raze();
  return ((0)); // expected-warning {{'return' will never be executed}}
}

void trivial_dead_return_void() {
  raze();
  return; // expected-warning {{'return' will never be executed}}
}

MyEnum trivial_dead_return_enum() {
  raze();
  return Value1; // expected-warning {{'return' will never be executed}}
}

MyEnum trivial_dead_return_enum_2(int x) {
  switch (x) {
    case 1: return 1;
    case 2: return 2;
    case 3: return 3;
    default: return 4;
  }

  return 2; // expected-warning {{will never be executed}}
}

const char *trivial_dead_return_cstr() {
  raze();
  return ""; // expected-warning {{return' will never be executed}}
}

char trivial_dead_return_char() {
  raze();
  return ' '; // expected-warning {{return' will never be executed}}
}

MyEnum nontrivial_dead_return_enum_2(int x) {
  switch (x) {
    case 1: return 1;
    case 2: return 2;
    case 3: return 3;
    default: return 4;
  }

  return calledFun(); // expected-warning {{will never be executed}}
}

enum X { A, B, C };

int covered_switch(enum X x) {
  switch (x) {
  case A: return 1;
  case B: return 2;
  case C: return 3;
  }
  return 4; // no-warning
}

// Test unreachable code depending on configuration values
#define CONFIG_CONSTANT 1
int test_config_constant(int x) {
  if (!CONFIG_CONSTANT) {
    calledFun(); // no-warning
    return 1;
  }
  if (!1) { // expected-note {{silence by adding parentheses to mark code as explicitly dead}}
    calledFun(); // expected-warning {{will never be executed}}
    return 1;
  }
  if (sizeof(int) > sizeof(char)) {
    calledFun(); // no-warning
    return 1;
  }
  if (x > 10)
    return CONFIG_CONSTANT ? calledFun() : calledFun(); // no-warning
  else
    return 1 ? // expected-note {{silence by adding parentheses to mark code as explicitly dead}}
      calledFun() :
      calledFun(); // expected-warning {{will never be executed}}
}

int sizeof_int(int x, int y) {
  if (sizeof(long) == sizeof(int))
    return 1; // no-warning
  if (sizeof(long) != sizeof(int))
    return 0; // no-warning
  if (x && y && sizeof(long) < sizeof(char))
    return 0; // no-warning
  return 2; // no-warning
}

enum MyEnum2 {
  ME_A = CONFIG_CONSTANT,
  ME_B = 1
};

int test_MyEnum() {
  if (!ME_A)
    return 1; // no-warning
  if (ME_A)
    return 2; // no-warning
  if (ME_B)
    return 3;
  if (!ME_B) // expected-warning {{will never be executed}}
    return 4; // expected-warning {{will never be executed}}
  return 5;
}

// Test for idiomatic do..while.
int test_do_while(int x) {
  do {
    if (x == calledFun())
      break;
    ++x;
    break;
  }
  while (0); // no-warning
  return x;
}

int test_do_while_nontrivial_cond(int x) {
  do {
    if (x == calledFun())
      break;
    ++x;
    break;
  }
  while (calledFun()); // expected-warning {{will never be executed}}
  return x;
}

// Diagnostic control: -Wunreachable-code-return.

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunreachable-code-return"

void trivial_dead_return_void_SUPPRESSED() {
  raze();
  return; // no-warning
}

MyEnum trivial_dead_return_enum_SUPPRESSED() {
  raze();
  return Value1; // no-warning
}

#pragma clang diagnostic pop

// Diagnostic control: -Wunreachable-code-break.

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunreachable-code-break"

int test_break_preceded_by_noreturn_SUPPRESSED(int i) {
  switch (i) {
    case 1:
      raze();
      break; // no-warning
    case 2:
      raze();
      break; // no-warning
      warn_here(); // expected-warning {{will never be executed}}
    case 3:
      return 1;
      break; // no-warning
    default:
      break;
      break; // no-warning
  }
  return i;
}

#pragma clang diagnostic pop

// Test "silencing" with parentheses.
void test_with_paren_silencing(int x) {
  if (0) calledFun(); // expected-warning {{will never be executed}} expected-note {{silence by adding parentheses to mark code as explicitly dead}}
  if ((0)) calledFun(); // no-warning

  if (1) // expected-note {{silence by adding parentheses to mark code as explicitly dead}}
    calledFun();
  else
    calledFun(); // expected-warning {{will never be executed}}

  if ((1))
    calledFun();
  else
    calledFun(); // no-warning
  
  if (!1) // expected-note {{silence by adding parentheses to mark code as explicitly dead}}
    calledFun(); // expected-warning {{code will never be executed}}
  else
    calledFun();
  
  if ((!1))
    calledFun(); // no-warning
  else
    calledFun();
  
  if (!(1))
    calledFun(); // no-warning
  else
    calledFun();
}

// rdar://24570531

struct StructWithPointer {
  void *p;
};

void emitJustOneWarningForOr(struct StructWithPointer *s) {
  if (1 || !s->p) // expected-note {{silence by adding parentheses to mark code as explicitly dead}}
    return; // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:"/* DISABLES CODE */ ("
            // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:8-[[@LINE-2]]:8}:")"
  emitJustOneWarningForOr(s); // expected-warning {{code will never be executed}}
}

void emitJustOneWarningForOrSilenced(struct StructWithPointer *s) {
  if ((1) || !s->p)
    return;

  emitJustOneWarningForOrSilenced(s); // no warning
}

void emitJustOneWarningForOr2(struct StructWithPointer *s) {
  if (1 || !s->p) // expected-warning {{code will never be executed}}
    return; // expected-note@-1 {{silence by adding parentheses to mark code as explicitly dead}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:7-[[@LINE-2]]:7}:"/* DISABLES CODE */ ("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:8-[[@LINE-3]]:8}:")"
}

void wrapOneInFixit(struct StructWithPointer *s) {
  if (!s->p || 1) // expected-note {{silence by adding parentheses to mark code as explicitly dead}}
    return; // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:16-[[@LINE-1]]:16}:"/* DISABLES CODE */ ("
            // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:17-[[@LINE-2]]:17}:")"
  wrapOneInFixit(s); // expected-warning {{code will never be executed}}
}

void unaryOpNoFixit() {
  if (- 1)
    return; // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]
  unaryOpNoFixit(); // expected-warning {{code will never be executed}}
}

void unaryOpStrictFixit(struct StructWithPointer *s) {
  if (!(s->p && 0)) // expected-note {{silence by adding parentheses to mark code as explicitly dead}}
    return; // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:17-[[@LINE-1]]:17}:"/* DISABLES CODE */ ("
            // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:18-[[@LINE-2]]:18}:")"
  unaryOpStrictFixit(s); // expected-warning {{code will never be executed}}
}

void unaryOpFixitCastSubExpr(int x) {
  if (! (int)0) // expected-note {{silence by adding parentheses to mark code as explicitly dead}}
    return; // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:7}:"/* DISABLES CODE */ ("
            // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:15-[[@LINE-2]]:15}:")"
  unaryOpFixitCastSubExpr(x); // expected-warning {{code will never be executed}}
}

#define false 0
#define true 1

void testTrueFalseMacros() {
  if (false) // expected-note {{silence by adding parentheses to mark code as explicitly dead}}
    testTrueFalseMacros(); // expected-warning {{code will never be executed}}
  if (!true) // expected-note {{silence by adding parentheses to mark code as explicitly dead}}
    testTrueFalseMacros(); // expected-warning {{code will never be executed}}
}

int pr13910_foo(int x) {
  if (x == 1)
    return 0;
  else
    return x;
  __builtin_unreachable(); // expected no warning
  __builtin_assume(0); // expected no warning
}

int pr13910_bar(int x) {
  switch (x) {
  default:
    return x + 1;
  }
  pr13910_foo(x); // expected-warning {{code will never be executed}}
}

int pr13910_bar2(int x) {
  if (x == 1)
    return 0;
  else
    return x;
  pr13910_foo(x);          // expected-warning {{code will never be executed}}
  __builtin_unreachable(); // expected no warning
  __builtin_assume(0);     // expected no warning
  pr13910_foo(x);          // expected-warning {{code will never be executed}}
}

void pr13910_noreturn() {
  raze();
  __builtin_unreachable(); // expected no warning
  __builtin_assume(0); // expected no warning
}

void pr13910_assert() {
  myassert(0 && "unreachable");
  return;
  __builtin_unreachable(); // expected no warning
  __builtin_assume(0); // expected no warning
}
