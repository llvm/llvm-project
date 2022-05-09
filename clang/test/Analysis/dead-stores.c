// RUN: %check_analyzer_fixit %s %t \
// RUN:   -Wunused-variable -fblocks -Wno-unreachable-code \
// RUN:   -analyzer-checker=core,deadcode.DeadStores \
// RUN:   -analyzer-config deadcode.DeadStores:ShowFixIts=true \
// RUN:   -analyzer-config \
// RUN:       deadcode.DeadStores:WarnForDeadNestedAssignments=false \
// RUN:   -verify=non-nested

// RUN: %check_analyzer_fixit %s %t \
// RUN:   -Wunused-variable -fblocks -Wno-unreachable-code \
// RUN:   -analyzer-checker=core,deadcode.DeadStores \
// RUN:   -analyzer-config deadcode.DeadStores:ShowFixIts=true \
// RUN:   -verify=non-nested,nested

extern int printf(const char *, ...);

void f1(void) {
  int k, y; // non-nested-warning {{unused variable 'k'}}
            // non-nested-warning@-1 {{unused variable 'y'}}
  int abc = 1;
  long idx = abc + 3 * 5; // non-nested-warning {{never read}}
                          // non-nested-warning@-1 {{unused variable 'idx'}}
  // CHECK-FIXES:      int abc = 1;
  // CHECK-FIXES-NEXT: long idx;
}

void f2(void *b) {
  char *c = (char *)b; // no-warning
  char *d = b + 1;     // non-nested-warning {{never read}}
                       // non-nested-warning@-1 {{unused variable 'd'}}
  // CHECK-FIXES:      char *c = (char *)b;
  // CHECK-FIXES-NEXT: char *d;

  printf("%s", c);
}

int f(void);
void f3(void) {
  int r;
  if ((r = f()) != 0) { // no-warning
    int y = r;          // no-warning
    printf("the error is: %d\n", y);
  }
}

void f4(int k) {
  k = 1;
  if (k)
    f1();
  k = 2; // non-nested-warning {{never read}}
}

void f5(void) {
  int x = 4;   // no-warning
  int *p = &x; // non-nested-warning {{never read}}
               // non-nested-warning@-1 {{unused variable 'p'}}
  // CHECK-FIXES:      int x = 4;
  // CHECK-FIXES-NEXT: int *p;
}

int f6(void) {
  int x = 4;
  ++x; // no-warning
  return 1;
}

int f7(int *p) {
  // This is allowed for defensive programming.
  p = 0; // no-warning
  return 1;
}

int f7b(int *p) {
  // This is allowed for defensive programming.
  p = (0); // no-warning
  return 1;
}

int f7c(int *p) {
  // This is allowed for defensive programming.
  p = (void *)0; // no-warning
  return 1;
}

int f7d(int *p) {
  // This is allowed for defensive programming.
  p = (void *)(0); // no-warning
  return 1;
}

// Warn for dead stores in nested expressions.
int f8(int *p) {
  extern int *baz(void);
  if ((p = baz())) // nested-warning {{Although the value stored}}
    return 1;
  return 0;
}

int f9(void) {
  int x = 4;
  x = x + 10; // non-nested-warning {{never read}}
  return 1;
}

int f10(void) {
  int x = 4;
  x = 10 + x; // non-nested-warning {{never read}}
  return 1;
}

int f11(void) {
  int x = 4;
  return x++; // non-nested-warning {{never read}}
}

int f11b(void) {
  int x = 4;
  return ((((++x)))); // no-warning
}

int f12a(int y) {
  int x = y; // non-nested-warning {{unused variable 'x'}}
  return 1;
}

int f12b(int y) {
  int x __attribute__((unused)) = y; // no-warning
  return 1;
}

int f12c(int y) {
  // Allow initialiation of scalar variables by parameters as a form of
  // defensive programming.
  int x = y; // no-warning
  x = 1;
  return x;
}

// Filed with PR 2630.  This code should produce no warnings.
int f13(void) {
  int a = 1;
  int b, c = b = a + a;

  if (b > 0)
    return (0);
  return (a + b + c);
}

// Filed with PR 2763.
int f14(int count) {
  int index, nextLineIndex;
  for (index = 0; index < count; index = nextLineIndex + 1) {
    nextLineIndex = index + 1; // no-warning
    continue;
  }
  return index;
}

// Test case for <rdar://problem/6248086>
void f15(unsigned x, unsigned y) {
  int count = x * y; // no-warning
  int z[count];      // non-nested-warning {{unused variable 'z'}}
}

// Warn for dead stores in nested expressions.
int f16(int x) {
  x = x * 2;
  x = sizeof(int[x = (x || x + 1) * 2]) ? 5 : 8;
  // nested-warning@-1 {{Although the value stored}}
  return x;
}

// Self-assignments should not be flagged as dead stores.
void f17(void) {
  int x = 1;
  x = x;
}

// <rdar://problem/6506065>
// The values of dead stores are only "consumed" in an enclosing expression
// what that value is actually used.  In other words, don't say "Although the
// value stored to 'x' is used...".
int f18(void) {
  int x = 0; // no-warning
  if (1)
    x = 10; // non-nested-warning {{Value stored to 'x' is never read}}
  while (1)
    x = 10; // non-nested-warning {{Value stored to 'x' is never read}}
  // unreachable.
  do
    x = 10; // no-warning
  while (1);
  return (x = 10); // no-warning
}

int f18_a(void) {
  int x = 0;       // no-warning
  return (x = 10); // nested-warning {{Although the value stored}}
}

void f18_b(void) {
  int x = 0; // no-warning
  if (1)
    x = 10; // non-nested-warning {{Value stored to 'x' is never read}}
}

void f18_c(void) {
  int x = 0;
  while (1)
    x = 10; // non-nested-warning {{Value stored to 'x' is never read}}
}

void f18_d(void) {
  int x = 0; // no-warning
  do
    x = 10; // non-nested-warning {{Value stored to 'x' is never read}}
  while (1);
}

// PR 3514: false positive `dead initialization` warning for init to global
//  http://llvm.org/bugs/show_bug.cgi?id=3514
extern const int MyConstant;
int f19(void) {
  int x = MyConstant; // no-warning
  x = 1;
  return x;
}

int f19b(void) { // This case is the same as f19.
  const int MyConstant = 0;
  int x = MyConstant; // no-warning
  x = 1;
  return x;
}

void f20(void) {
  int x = 1; // no-warning
#pragma unused(x)
}

void halt(void) __attribute__((noreturn));
int f21(void) {
  int x = 4;
  x = x + 1; // non-nested-warning {{never read}}
  if (1) {
    halt();
    (void)x;
  }
  return 1;
}

int j;
void f22(void) {
  int x = 4;
  int y1 = 4;
  int y2 = 4;
  int y3 = 4;
  int y4 = 4;
  int y5 = 4;
  int y6 = 4;
  int y7 = 4;
  int y8 = 4;
  int y9 = 4;
  int y10 = 4;
  int y11 = 4;
  int y12 = 4;
  int y13 = 4;
  int y14 = 4;
  int y15 = 4;
  int y16 = 4;
  int y17 = 4;
  int y18 = 4;
  int y19 = 4;
  int y20 = 4;

  x = x + 1; // non-nested-warning {{never read}}
  ++y1;
  ++y2;
  ++y3;
  ++y4;
  ++y5;
  ++y6;
  ++y7;
  ++y8;
  ++y9;
  ++y10;
  ++y11;
  ++y12;
  ++y13;
  ++y14;
  ++y15;
  ++y16;
  ++y17;
  ++y18;
  ++y19;
  ++y20;

  switch (j) {
  case 1:
    if (0)
      (void)x;
    if (1) {
      (void)y1;
      return;
    }
    (void)x;
    break;
  case 2:
    if (0)
      (void)x;
    else {
      (void)y2;
      return;
    }
    (void)x;
    break;
  case 3:
    if (1) {
      (void)y3;
      return;
    } else
      (void)x;
    (void)x;
    break;
  case 4:
    0 ?: ((void)y4, ({ return; }));
    (void)x;
    break;
  case 5:
    1 ?: (void)x;
    0 ? (void)x : ((void)y5, ({ return; }));
    (void)x;
    break;
  case 6:
    1 ? ((void)y6, ({ return; })) : (void)x;
    (void)x;
    break;
  case 7:
    (void)(0 && x);
    (void)y7;
    (void)(0 || (y8, ({ return; }), 1));
    // non-nested-warning@-1 {{left operand of comma operator has no effect}}
    (void)x;
    break;
  case 8:
    (void)(1 && (y9, ({ return; }), 1));
    // non-nested-warning@-1 {{left operand of comma operator has no effect}}
    (void)x;
    break;
  case 9:
    (void)(1 || x);
    (void)y10;
    break;
  case 10:
    while (0) {
      (void)x;
    }
    (void)y11;
    break;
  case 11:
    while (1) {
      (void)y12;
    }
    (void)x;
    break;
  case 12:
    do {
      (void)y13;
    } while (0);
    (void)y14;
    break;
  case 13:
    do {
      (void)y15;
    } while (1);
    (void)x;
    break;
  case 14:
    for (;;) {
      (void)y16;
    }
    (void)x;
    break;
  case 15:
    for (; 1;) {
      (void)y17;
    }
    (void)x;
    break;
  case 16:
    for (; 0;) {
      (void)x;
    }
    (void)y18;
    break;
  case 17:
    __builtin_choose_expr(0, (void)x, ((void)y19, ({ return; })));
    (void)x;
    break;
  case 19:
    __builtin_choose_expr(1, ((void)y20, ({ return; })), (void)x);
    (void)x;
    break;
  }
}

void f23_aux(const char *s);
void f23(int argc, char **argv) {
  int shouldLog = (argc > 1); // no-warning
  ^{
    if (shouldLog)
      f23_aux("I did too use it!\n");
    else
      f23_aux("I shouldn't log.  Wait.. d'oh!\n");
  }();
}

void f23_pos(int argc, char **argv) {
  int shouldLog = (argc > 1);
  // non-nested-warning@-1 {{Value stored to 'shouldLog' during its initialization is never read}}
  // non-nested-warning@-2 {{unused variable 'shouldLog'}}
  // CHECK-FIXES:      void f23_pos(int argc, char **argv) {
  // CHECK-FIXES-NEXT:   int shouldLog;
  ^{
    f23_aux("I did too use it!\n");
  }();
}

void f24_A(int y) {
  // FIXME: One day this should be reported as dead since 'z = x + y' is dead.
  int x = (y > 2); // no-warning
  ^{
    int z = x + y;
    // non-nested-warning@-1 {{Value stored to 'z' during its initialization is never read}}
    // non-nested-warning@-2 {{unused variable 'z'}}
    // CHECK-FIXES:      void f24_A(int y) {
    // CHECK-FIXES-NEXT:   //
    // CHECK-FIXES-NEXT:   int x = (y > 2);
    // CHECK-FIXES-NEXT:   ^{
    // CHECK-FIXES-NEXT:     int z;
  }();
}

void f24_B(int y) {
  // FIXME: One day this should be reported as dead since 'x' is just overwritten.
  __block int x = (y > 2); // no-warning
  ^{
    // FIXME: This should eventually be a dead store since it is never read either.
    x = 5; // no-warning
  }();
}

int f24_C(int y) {
  // FIXME: One day this should be reported as dead since 'x' is just overwritten.
  __block int x = (y > 2); // no-warning
  ^{
    x = 5; // no-warning
  }();
  return x;
}

int f24_D(int y) {
  __block int x = (y > 2); // no-warning
  ^{
    if (y > 4)
      x = 5; // no-warning
  }();
  return x;
}

// This example shows that writing to a variable captured by a block means that
// it might not be dead.
int f25(int y) {
  __block int x = (y > 2);
  __block int z = 0;
  void (^foo)(void) = ^{
    z = x + y;
  };
  x = 4; // no-warning
  foo();
  return z;
}

// This test is mostly the same as 'f25', but shows that the heuristic of
// pruning out dead stores for variables that are just marked '__block' is
// overly conservative.
int f25_b(int y) {
  // FIXME: we should eventually report a dead store here.
  __block int x = (y > 2);
  __block int z = 0;
  x = 4; // no-warning
  return z;
}

int f26_nestedblocks(void) {
  int z;
  z = 1;
  __block int y = 0;
  ^{
    int k;
    k = 1; // non-nested-warning {{Value stored to 'k' is never read}}
    ^{
      y = z + 1;
    }();
  }();
  return y;
}

// The FOREACH macro in QT uses 'break' statements within statement expressions
// placed within the increment code of for loops.
void rdar8014335(void) {
  for (int i = 0 ; i != 10 ; ({ break; })) {
    for (;; ({ ++i; break; }))
      ;
    // non-nested-warning@-2 {{'break' is bound to current loop, GCC binds it to the enclosing loop}}
    // Note that the next value stored to 'i' is never executed
    // because the next statement to be executed is the 'break'
    // in the increment code of the first loop.
    i = i * 3; // non-nested-warning {{Value stored to 'i' is never read}}
  }
}

// <rdar://problem/8320674> NullStmts followed by do...while() can lead to disconnected CFG
//
// This previously caused bogus dead-stores warnings because the body of the first do...while was
// disconnected from the entry of the function.
typedef struct { float r; float i; } s_rdar8320674;
typedef struct { s_rdar8320674 x[1]; } s2_rdar8320674;

void rdar8320674(s_rdar8320674 *z, unsigned y, s2_rdar8320674 *st, int m)
{
    s_rdar8320674 * z2;
    s_rdar8320674 * tw1 = st->x;
    s_rdar8320674 t;
    z2 = z + m;
    do{
        ; ;
        do{ (t).r = (*z2).r*(*tw1).r - (*z2).i*(*tw1).i; (t).i = (*z2).r*(*tw1).i + (*z2).i*(*tw1).r; }while(0);
        tw1 += y;
        do { (*z2).r=(*z).r-(t).r; (*z2).i=(*z).i-(t).i; }while(0);
        do { (*z).r += (t).r; (*z).i += (t).i; }while(0);
        ++z2;
        ++z;
    }while (--m);
}

// Avoid dead stores resulting from an assignment (and use) being unreachable.
void rdar8405222_aux(int i);
void rdar8405222(void) {
  const int show = 0;
  int i = 0;
  if (show)
    i = 5; // no-warning
  if (show)
    rdar8405222_aux(i);
}

// Look through chains of assignments, e.g.: int x = y = 0, when employing
// silencing heuristics.
int radar11185138_foo(void) {
  int x, y;
  x = y = 0; // non-nested-warning {{never read}}
  return y;
}

int rdar11185138_bar(void) {
  int y;
  int x = y = 0; // nested-warning {{Although the value stored}}
  x = 2;
  y = 2;
  return x + y;
}

int *radar11185138_baz(void) {
  int *x, *y;
  x = y = 0; // no-warning
  return y;
}

int getInt(void);
int *getPtr(void);
void testBOComma(void) {
  int x0 = (getInt(), 0); // non-nested-warning {{unused variable 'x0'}}
  int x1 = (getInt(), getInt());
  // non-nested-warning@-1 {{Value stored to 'x1' during its initialization is never read}}
  // non-nested-warning@-2 {{unused variable 'x1'}}

  int x2 = (getInt(), getInt(), getInt());
  // non-nested-warning@-1 {{Value stored to 'x2' during its initialization is never read}}
  // non-nested-warning@-2 {{unused variable 'x2'}}

  int x3;
  x3 = (getInt(), getInt(), 0);
  // non-nested-warning@-1 {{Value stored to 'x3' is never read}}

  int x4 = (getInt(), (getInt(), 0));
  // non-nested-warning@-1 {{unused variable 'x4'}}

  int y;
  int x5 = (getInt(), (y = 0));
  // non-nested-warning@-1 {{unused variable 'x5'}}
  // nested-warning@-2 {{Although the value stored}}

  int x6 = (getInt(), (y = getInt()));
  // non-nested-warning@-1 {{Value stored to 'x6' during its initialization is never read}}
  // non-nested-warning@-2 {{unused variable 'x6'}}
  // nested-warning@-3 {{Although the value stored}}

  int x7 = 0, x8 = getInt();
  // non-nested-warning@-1 {{Value stored to 'x8' during its initialization is never read}}
  // non-nested-warning@-2 {{unused variable 'x8'}}
  // non-nested-warning@-3 {{unused variable 'x7'}}

  int x9 = getInt(), x10 = 0;
  // non-nested-warning@-1 {{Value stored to 'x9' during its initialization is never read}}
  // non-nested-warning@-2 {{unused variable 'x9'}}
  // non-nested-warning@-3 {{unused variable 'x10'}}

  int m = getInt(), mm, mmm;
  // non-nested-warning@-1 {{Value stored to 'm' during its initialization is never read}}
  // non-nested-warning@-2 {{unused variable 'm'}}
  // non-nested-warning@-3 {{unused variable 'mm'}}
  // non-nested-warning@-4 {{unused variable 'mmm'}}

  int n, nn = getInt();
  // non-nested-warning@-1 {{Value stored to 'nn' during its initialization is never read}}
  // non-nested-warning@-2 {{unused variable 'n'}}
  // non-nested-warning@-3 {{unused variable 'nn'}}

  int *p;
  p = (getPtr(), (int *)0); // no warning
}

void testVolatile(void) {
  volatile int v;
  v = 0; // no warning
}

struct Foo {
  int x;
  int y;
};

struct Foo rdar34122265_getFoo(void);

int rdar34122265_test(int input) {
  // This is allowed for defensive programming.
  struct Foo foo = {0, 0};
  if (input > 0) {
    foo = rdar34122265_getFoo();
  } else {
    return 0;
  }
  return foo.x + foo.y;
}

void rdar34122265_test_cast(void) {
  // This is allowed for defensive programming.
  struct Foo foo = {0, 0};
  (void)foo;
}

struct Bar {
  struct Foo x, y;
};

struct Bar rdar34122265_getBar(void);

int rdar34122265_test_nested(int input) {
  // This is allowed for defensive programming.
  struct Bar bar = {{0, 0}, {0, 0}};
  if (input > 0) {
    bar = rdar34122265_getBar();
  } else {
    return 0;
  }
  return bar.x.x + bar.y.y;
}
