// RUN: %clang_cc1 -fsyntax-only -Wno-strict-prototypes -verify %s -fblocks
// RUN: %clang_cc1 -fsyntax-only -Wno-strict-prototypes -verify %s -fblocks -fexperimental-new-constant-interpreter
void donotwarn(void);

int (^IFP) ();
int (^II) (int);
int test1(void) {
  int (^PFR) (int) = 0; // OK
  PFR = II;             // OK

  if (PFR == II)        // OK
    donotwarn();

  if (PFR == IFP)       // OK
    donotwarn();

  if (PFR == (int (^) (int))IFP) // OK
    donotwarn();

  if (PFR == 0)         // OK
    donotwarn();

  if (PFR)              // OK
    donotwarn();

  if (!PFR)             // OK
    donotwarn();

  return PFR != IFP;    // OK
}

int test2(double (^S)()) {
  double (^I)(int)  = (void*) S;
  (void*)I = (void *)S; // expected-error {{assignment to cast is illegal, lvalue casts are not supported}}

  void *pv = I;

  pv = S;

  I(1);

  return (void*)I == (void *)S;
}

int^ x; // expected-error {{block pointer to non-function type is invalid}}
int^^ x1; // expected-error {{block pointer to non-function type is invalid}} expected-error {{block pointer to non-function type is invalid}}

void test3(void) {
  char *^ y; // expected-error {{block pointer to non-function type is invalid}}
}



enum {NSBIRLazilyAllocated = 0};

int test4(int argc) {
  ^{
    switch (argc) {
      case NSBIRLazilyAllocated:  // is an integer constant expression.
      default:
        break;
    }
  }();
  return 0;
}


void bar(void*);
static int test5g;
void test5() {
  bar(^{ test5g = 1; });
}

const char*test6(void) {
  return ^{
    return __func__;
  } ();
}

void (^test7a)();
int test7(void (^p)()) {
  return test7a == p;
}


void test8(void) {
somelabel:
  ^{ goto somelabel; }();   // expected-error {{use of undeclared label 'somelabel'}}
}

void test9(void) {
  goto somelabel;       // expected-error {{use of undeclared label 'somelabel'}}
  ^{ somelabel: ; }();
}

void test10(int i) {
  switch (i) {
  case 41: ;
  ^{ case 42: ; }();     // expected-error {{'case' statement not in switch statement}}
  }
}

void test11(int i) {
  switch (i) {
  case 41: ;
    ^{ break; }();     // expected-error {{'break' statement not in loop or switch statement}}
  }
  
  for (; i < 100; ++i)
    ^{ break; }();     // expected-error {{'break' statement not in loop or switch statement}}
}

void (^test12f)(void);
void test12() {
  test12f = ^test12f;  // expected-error {{type name requires a specifier or qualifier}} expected-error {{expected expression}}
}

void *test13 = ^{
  int X = 32;

  void *P = ^{
    return X+4;  // References outer block's "X", so outer block is constant.
  };
};

void test14(void) {
  int X = 32;
  static void *P = ^{  // expected-error {{initializer element is not a compile-time constant}}

    void *Q = ^{
      // References test14's "X": outer block is non-constant.
      return X+4;
    };
  };
}

enum { LESS };

void foo(long (^comp)()) { // expected-note{{passing argument to parameter 'comp' here}}
}

void (^test15f)(void);
void test15(void) {
  foo(^{ return LESS; }); // expected-error {{incompatible block pointer types passing 'int (^)(void)' to parameter of type 'long (^)()'}}
}

__block int test16i;  // expected-error {{__block attribute not allowed, only allowed on local variables}}

void test16(__block int i) { // expected-error {{__block attribute not allowed, only allowed on local variables}}
  int size = 5;
  extern __block double extern_var; // expected-error {{__block attribute not allowed, only allowed on local variables}}
  static __block char * pch; // expected-error {{__block attribute not allowed, only allowed on local variables}}
  __block int a[size]; // expected-error {{__block attribute not allowed on declaration with a variably modified type}}
  __block int (*ap)[size]; // expected-error {{__block attribute not allowed on declaration with a variably modified type}}
}

void f();

void test17(void) {
  void (^bp)(int);
  void (*rp)(int);
  void (^bp1)();
  void *vp = bp;

  f(1 ? bp : vp);
  f(1 ? vp : bp);
  f(1 ? bp : bp1);
  (void)(bp > rp); // expected-error {{invalid operands}}
  (void)(bp > 0); // expected-error {{invalid operands}}
  (void)(bp > bp); // expected-error {{invalid operands}}
  (void)(bp > vp); // expected-error {{invalid operands}}
  f(1 ? bp : rp); // expected-error {{incompatible operand types ('void (^)(int)' and 'void (*)(int)')}}
  (void)(bp == 1); // expected-error {{invalid operands to binary expression}}
  (void)(bp == 0);
  (void)(1 == bp); // expected-error {{invalid operands to binary expression}}
  (void)(0 == bp);
  (void)(bp < 1); // expected-error {{invalid operands to binary expression}}
  (void)(bp < 0); // expected-error {{invalid operands to binary expression}}
  (void)(1 < bp); // expected-error {{invalid operands to binary expression}}
  (void)(0 < bp); // expected-error {{invalid operands to binary expression}}
}

void test18(void) {
  void (^const  blockA)(void) = ^{ };  // expected-note {{variable 'blockA' declared const here}}
  blockA = ^{ }; // expected-error {{cannot assign to variable 'blockA' with const-qualified type 'void (^const)(void)}}
}

int test19(void) {
  goto L0;       // expected-error {{cannot jump}}
  
  __block int x; // expected-note {{jump bypasses setup of __block variable}}
L0:
  x = 0;
  ^(){ ++x; }();
  return x;
}

void test20(void) {
  int n = 7;
  int vla[n]; // expected-note {{declared here}}
  int (*vm)[n] = 0; // expected-note {{declared here}}
  vla[1] = 4341;
  ^{
    (void)vla[1];  // expected-error {{cannot refer to declaration with a variably modified type inside block}}
    (void)(vm+1);  // expected-error {{cannot refer to declaration with a variably modified type inside block}}
  }();
}

void test21(void) {
  int a[7]; // expected-note {{declared here}}
  __block int b[10]; // expected-note {{declared here}}
  a[1] = 1;
  ^{
    (void)a[1]; // expected-error {{cannot refer to declaration with an array type inside block}}
    (void)b[1]; // expected-error {{cannot refer to declaration with an array type inside block}}
  }();
}

const char * (^func)(void) = ^{ return __func__; };
const char * (^function)(void) = ^{ return __FUNCTION__; };
const char * (^pretty)(void) = ^{ return __PRETTY_FUNCTION__; };
