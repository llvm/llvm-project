// RUN: %clang_cc1 -triple arm64-apple-ios -DIS_DARWIN -std=c23 -fsyntax-only -verify -fptrauth-intrinsics %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -std=c23 -fsyntax-only -verify -fptrauth-intrinsics %s

#if defined(IS_DARWIN) && !__has_extension(ptrauth_qualifier)
// This error means that the __ptrauth qualifier availability test says  that it
// is not available. This error is not expected in the output, if it is seen
// there is a feature detection regression.
#error __ptrauth qualifier not enabled
#endif

#if defined(IS_DARWIN) && !__has_feature(ptrauth_qualifier)
// This error means that the __has_feature test for ptrauth_qualifier has
// failed, despite it being expected on darwin.
#error __ptrauth qualifier not enabled
#elif !defined(IS_DARWIN) && (__has_feature(ptrauth_qualifier) || __has_extension(ptrauth_qualifier))
#error ptrauth_qualifier labeled a feature on a non-darwin platform
#endif

#if !defined (__PTRAUTH__)
#error __PTRAUTH__ test macro not defined when ptrauth is enabled
#endif

#if __aarch64__
#define VALID_CODE_KEY 0
#define VALID_DATA_KEY 2
#define INVALID_KEY 200
#else
#error Provide these constants if you port this test
#endif

int * __ptrauth(VALID_DATA_KEY) valid0;
int *ptr0;

typedef int *intp;

int nonConstantGlobal = 5;

__ptrauth(INVALID_KEY) int invalid2; // expected-error{{200 does not identify a valid pointer authentication key for the current target}}
__ptrauth(VALID_DATA_KEY) int invalid3; // expected-error {{'__ptrauth' qualifier only applies to pointer or pointer sized integer types; 'int' is invalid}}
__ptrauth(VALID_DATA_KEY) int *invalid4; // expected-error {{'__ptrauth' qualifier only applies to pointer or pointer sized integer types; 'int' is invalid}}
int * (__ptrauth(VALID_DATA_KEY) invalid5); // expected-error{{expected identifier or '('}} expected-error{{expected ')'}} expected-note {{to match this '('}}
int *__ptrauth(VALID_DATA_KEY) __ptrauth(VALID_DATA_KEY) invalid6; // expected-error{{type 'int *__ptrauth(2,0,0)' is already '__ptrauth'-qualified}}
int * __ptrauth(VALID_DATA_KEY, 2) invalid7; // expected-error {{invalid address discrimination flag '2'; '__ptrauth' requires '0' or '1'}}
int * __ptrauth(VALID_DATA_KEY, -1) invalid8; // expected-error {{invalid address discrimination flag '-1'; '__ptrauth' requires '0' or '1'}}
int * __ptrauth(VALID_DATA_KEY, 1, -1) invalid9; // expected-error {{invalid extra discriminator flag '-1'; '__ptrauth' requires a value between '0' and '65535'}}
int * __ptrauth(VALID_DATA_KEY, 1, 100000) invalid10; // expected-error {{invalid extra discriminator flag '100000'; '__ptrauth' requires a value between '0' and '65535'}}
int * __ptrauth(VALID_DATA_KEY, 1, nonConstantGlobal) invalid12; // expected-error {{argument to '__ptrauth' must be an integer constant expression}}
int * __ptrauth(VALID_DATA_KEY, nonConstantGlobal, 1000) invalid13; // expected-error {{argument to '__ptrauth' must be an integer constant expression}}
int * __ptrauth(nonConstantGlobal, 1, 1000) invalid14; // expected-error{{expression is not an integer constant expression}}

int * __ptrauth(VALID_DATA_KEY) valid0;
int * __ptrauth(VALID_DATA_KEY) *valid1;
__ptrauth(VALID_DATA_KEY) intp valid2;
__ptrauth(VALID_DATA_KEY) intp *valid3;
intp __ptrauth(VALID_DATA_KEY) valid4;
intp __ptrauth(VALID_DATA_KEY) *valid5;
int * __ptrauth(VALID_DATA_KEY, 0) valid6;
int * __ptrauth(VALID_DATA_KEY, 1) valid7;
int * __ptrauth(VALID_DATA_KEY, (_Bool) 1) valid8;
int * __ptrauth(VALID_DATA_KEY, 1, 0) valid9;
int * __ptrauth(VALID_DATA_KEY, 1, 65535) valid10;

int * __ptrauth(VALID_DATA_KEY) array0[10];
int (* __ptrauth(VALID_DATA_KEY) array1)[10];

extern intp redeclaration0; // expected-note {{previous declaration}}
extern intp __ptrauth(VALID_DATA_KEY) redeclaration0; // expected-error{{redeclaration of 'redeclaration0' with a different type: '__ptrauth(2,0,0) intp' (aka 'int *__ptrauth(2,0,0)') vs 'intp' (aka 'int *')}}

extern intp redeclaration1; // expected-note {{previous declaration}}
extern intp __ptrauth(VALID_DATA_KEY) redeclaration1; // expected-error{{redeclaration of 'redeclaration1' with a different type: '__ptrauth(2,0,0) intp' (aka 'int *__ptrauth(2,0,0)') vs 'intp' (aka 'int *')}}

intp __ptrauth(VALID_DATA_KEY) redeclaration2; // expected-note {{previous definition}}
intp redeclaration2 = 0;                       // expected-error{{redefinition of 'redeclaration2' with a different type: 'intp' (aka 'int *') vs '__ptrauth(2,0,0) intp' (aka 'int *__ptrauth(2,0,0)')}}

intp __ptrauth(VALID_DATA_KEY) redeclaration3; // expected-note {{previous definition}}
intp redeclaration3 = 0;                       // expected-error{{redefinition of 'redeclaration3' with a different type: 'intp' (aka 'int *') vs '__ptrauth(2,0,0) intp' (aka 'int *__ptrauth(2,0,0)')}}

void illegal0(intp __ptrauth(VALID_DATA_KEY)); // expected-error {{parameter type may not be qualified with '__ptrauth'; type is '__ptrauth(2,0,0) intp' (aka 'int *__ptrauth(2,0,0)')}}
intp __ptrauth(VALID_DATA_KEY) illegal1(void); // expected-error {{return type may not be qualified with '__ptrauth'; type is '__ptrauth(2,0,0) intp' (aka 'int *__ptrauth(2,0,0)')}}

static_assert(_Generic(typeof(valid0), int * __ptrauth(VALID_DATA_KEY) : 1, int * : 0, default : 0));
static_assert(_Generic(typeof(valid0), int * __ptrauth(VALID_CODE_KEY) : 0, default : 1));
static_assert(_Generic(typeof_unqual(valid0), int * __ptrauth(VALID_DATA_KEY) : 0, int * : 1, default : 0));
static_assert(_Generic(valid0, int * __ptrauth(VALID_DATA_KEY) : 0, int * : 1, default : 0)); // expected-warning {{association of type 'int *__ptrauth(2,0,0)' will never be selected}}

static_assert(_Generic(array0, int * __ptrauth(VALID_DATA_KEY) * : 1, default : 0));
static_assert(_Generic(*array1, int * : 1, default : 0));

void test_code(intp p) {
  p = (intp __ptrauth(VALID_DATA_KEY)) 0; // expected-error {{cannot cast to '__ptrauth'-qualified type '__ptrauth(2,0,0) intp' (aka 'int *__ptrauth(2,0,0)')}}

  __ptrauth(VALID_DATA_KEY) intp pSpecial = p;
  pSpecial = p;
  intp pNormal = pSpecial;
  pNormal = pSpecial;

  intp __ptrauth(VALID_DATA_KEY) *ppSpecial0 = &pSpecial;
  intp __ptrauth(VALID_DATA_KEY) *ppSpecial1 = &pNormal; // expected-error {{initializing '__ptrauth(2,0,0) intp *' (aka 'int *__ptrauth(2,0,0) *') with an expression of type 'intp *' (aka 'int **') changes pointer authentication of pointee type}}
  intp *ppNormal0 = &pSpecial; // expected-error {{initializing 'intp *' (aka 'int **') with an expression of type '__ptrauth(2,0,0) intp *' (aka 'int *__ptrauth(2,0,0) *') changes pointer authentication of pointee type}}
  intp *ppNormal1 = &pNormal;

  intp *pp5 = (p ? &pSpecial : &pNormal); // expected-error {{'__ptrauth' qualification mismatch ('__ptrauth(2,0,0) intp *' (aka 'int *__ptrauth(2,0,0) *') and 'intp *' (aka 'int **'))}}
}

void test_array(void) {
  intp __ptrauth(VALID_DATA_KEY) pSpecialArray[10];
  intp __ptrauth(VALID_DATA_KEY) *ppSpecial0 = pSpecialArray;
  intp __ptrauth(VALID_DATA_KEY) *ppSpecial1 = &pSpecialArray[0];
}

__attribute__((overloadable)) int overload_func(int **);
__attribute__((overloadable)) float overload_func(int * __ptrauth(VALID_DATA_KEY) *);

static_assert(_Generic(typeof(overload_func(&ptr0)), int : 1, default : 0));
static_assert(_Generic(typeof(overload_func(&valid0)), float : 1, default : 0));

void func(int array[__ptrauth(VALID_DATA_KEY) 10]); // expected-error {{'__ptrauth' qualifier only applies to pointer or pointer sized integer types; 'int[10]' is invalid}}

struct S0 { // expected-note 4 {{struct S0' has subobjects that are non-trivial to copy}}
  intp __ptrauth(1, 1, 50) f0; // expected-note 4 {{f0 has type '__ptrauth(1,1,50) intp' (aka 'int *__ptrauth(1,1,50)') that is non-trivial to copy}}
};

union U0 { // expected-note 4 {{union U0' has subobjects that are non-trivial to copy}}
  struct S0 s0;
};

struct S1 {
  intp __ptrauth(1, 0, 50) f0;
};

union U1 {
  struct S1 s1;
};

union U2 { // expected-note 2 {{union U2' has subobjects that are non-trivial to copy}}
  intp __ptrauth(1, 1, 50) f0; // expected-note 2 {{f0 has type '__ptrauth(1,1,50) intp' (aka 'int *__ptrauth(1,1,50)') that is non-trivial to copy}}
  intp __ptrauth(1, 0, 50) f1;
};

struct S2 { // expected-note 2 {{struct S2' has subobjects that are non-trivial to copy}}
  intp __ptrauth(1, 1, 50) f0[4]; // expected-note 2 {{f0 has type '__ptrauth(1,1,50) intp' (aka 'int *__ptrauth(1,1,50)') that is non-trivial to copy}}
};

union U3 { // expected-note 2 {{union U3' has subobjects that are non-trivial to copy}}
  struct S2 s2;
};

struct S4 {
  union U0 u0;
};

union U0 foo0(union U0); // expected-error {{cannot use type 'union U0' for function/method return since it is a union that is non-trivial to copy}} expected-error {{cannot use type 'union U0' for a function/method parameter since it is a union that is non-trivial to copy}}
union U1 foo1(union U1);

union U2 foo2(union U2); // expected-error {{cannot use type 'union U2' for function/method return since it is a union that is non-trivial to copy}} expected-error {{cannot use type 'union U2' for a function/method parameter since it is a union that is non-trivial to copy}}

union U3 foo3(union U3); // expected-error {{cannot use type 'union U3' for function/method return since it is a union that is non-trivial to copy}} expected-error {{cannot use type 'union U3' for a function/method parameter since it is a union that is non-trivial to copy}}

struct S4 foo4(struct S4);  // expected-error {{cannot use type 'struct S4' for function/method return since it contains a union that is non-trivial to copy}} expected-error {{cannot use type 'struct S4' for a function/method parameter since it contains a union that is non-trivial to copy}}
