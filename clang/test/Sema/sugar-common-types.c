// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c99 -triple aarch64-arm-none-eabi -target-feature +bf16 -target-feature +sve

typedef struct N {} N;

typedef int B1;
typedef B1 X1;
typedef B1 Y1;

typedef void B2;
typedef B2 X2;
typedef B2 Y2;

typedef struct B3 {} B3;
typedef B3 X3;
typedef B3 Y3;

typedef struct B4 {} *B4;
typedef B4 X4;
typedef B4 Y4;

typedef __bf16 B5;
typedef B5 X5;
typedef B5 Y5;

typedef __SVInt8_t B6;
typedef B6 X6;
typedef B6 Y6;

N t1 = 0 ? (X1)0 : (Y1)0;   // expected-error {{incompatible type 'B1'}}
N t2 = 0 ? (X2)0 : 0;       // expected-error {{incompatible type 'X2'}}
N t3 = 0 ? 0 : (Y2)0;       // expected-error {{incompatible type 'Y2'}}
N t4 = 0 ? (X2)0 : (Y2)0;   // expected-error {{incompatible type 'B2'}}
N t5 = 0 ? (X3){} : (Y3){}; // expected-error {{incompatible type 'B3'}}
N t6 = 0 ? (X4)0 : (Y4)0;   // expected-error {{incompatible type 'B4'}}

X5 x5;
Y5 y5;
N t7 = 0 ? x5 : y5; // expected-error {{incompatible type 'B5'}}

void f8() {
  X6 x6;
  Y6 y6;
  N t8 = 0 ? x6 : y6; // expected-error {{incompatible type 'B6'}}
}
