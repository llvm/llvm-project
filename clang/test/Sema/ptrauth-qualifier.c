// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -verify -fptrauth-intrinsics %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fsyntax-only -verify -fptrauth-intrinsics %s

#if __has_feature(ptrauth_qualifier)
#warning __ptrauth qualifier enabled!
// expected-warning@-1 {{__ptrauth qualifier enabled!}}
#endif

#if __aarch64__
#define VALID_CODE_KEY 0
#define VALID_DATA_KEY 2
#define INVALID_KEY 200
#else
#error Provide these constants if you port this test
#endif

int * __ptrauth(VALID_DATA_KEY) valid0;

typedef int *intp;

int nonConstantGlobal = 5;

__ptrauth int invalid0; // expected-error{{expected '('}}
__ptrauth() int invalid1; // expected-error{{expected expression}}
__ptrauth(INVALID_KEY) int invalid2; // expected-error{{200 does not identify a valid pointer authentication key for the current target}}
__ptrauth(VALID_DATA_KEY) int invalid3; // expected-error {{'__ptrauth' qualifier only applies to pointer types; 'int' is invalid}}
__ptrauth(VALID_DATA_KEY) int *invalid4; // expected-error {{'__ptrauth' qualifier only applies to pointer types; 'int' is invalid}}
int * (__ptrauth(VALID_DATA_KEY) invalid5); // expected-error{{expected identifier or '('}} expected-error{{expected ')'}} expected-note {{to match this '('}}
int *__ptrauth(VALID_DATA_KEY) __ptrauth(VALID_DATA_KEY) invalid6; // expected-error{{type 'int *__ptrauth(2,0,0)' is already __ptrauth-qualified}}
int * __ptrauth(VALID_DATA_KEY, 2) invalid7; // expected-error {{invalid address discrimination flag '2'; '__ptrauth' requires '0' or '1'}}
int * __ptrauth(VALID_DATA_KEY, -1) invalid8; // expected-error {{invalid address discrimination flag '-1'; '__ptrauth' requires '0' or '1'}}
int * __ptrauth(VALID_DATA_KEY, 1, -1) invalid9; // expected-error {{invalid extra discriminator flag '-1'; '__ptrauth' requires a value between '0' and '65535'}}
int * __ptrauth(VALID_DATA_KEY, 1, 100000) invalid10; // expected-error {{invalid extra discriminator flag '100000'; '__ptrauth' requires a value between '0' and '65535'}}
int * __ptrauth(VALID_DATA_KEY, 1, nonConstantGlobal) invalid12; // expected-error {{argument to '__ptrauth' must be an integer constant expression}}
int * __ptrauth(VALID_DATA_KEY, nonConstantGlobal, 1000) invalid13; // expected-error {{argument to '__ptrauth' must be an integer constant expression}}
int * __ptrauth(nonConstantGlobal, 1, 1000) invalid14; // expected-error{{expression is not an integer constant expression}}
int * __ptrauth(VALID_DATA_KEY, 1, 1000, 12) invalid15; // expected-error{{qualifier must take between 1 and 3 arguments}}

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
