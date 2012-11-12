// RUN: %clang %s -fsyntax-only -Xclang -verify -fblocks -Wunreachable-code -Wno-unused-value -Wno-covered-switch-default

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
    i           // expected-warning {{will never be executed}}
      +=
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
      a[halt()  // expected-warning {{will never be executed}}
        ];
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
      int i = 0; // expected-warning{{will never be executed}}
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
void calledFun();
void test_mul_and_zero(int x) {
  if (x & 0) calledFun(); // expected-warning {{will never be executed}}
  if (0 & x) calledFun(); // expected-warning {{will never be executed}}
  if (x * 0) calledFun(); // expected-warning {{will never be executed}}
  if (0 * x) calledFun(); // expected-warning {{will never be executed}}
}
