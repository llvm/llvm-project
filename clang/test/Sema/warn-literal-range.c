// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c99 -verify %s

float f0 = 0.42f; // no-warning

float f1 = 1.4E-46f; // expected-warning {{magnitude of floating-point constant too small for type 'float'; minimum is 1.40129846E-45}}

float f2 = 3.4E+39f; // expected-warning {{magnitude of floating-point constant too large for type 'float'; maximum is 3.40282347E+38}}

float f3 = 0x4.2p42f; // no-warning

float f4 = 0x0.42p-1000f; // expected-warning {{magnitude of floating-point constant too small for type 'float'; minimum is 1.40129846E-45}}

float f5 = 0x0.42p+1000f; // expected-warning {{magnitude of floating-point constant too large for type 'float'; maximum is 3.40282347E+38}}

double d0 = 0.42; // no-warning

double d1 = 3.6E-4952; // expected-warning {{magnitude of floating-point constant too small for type 'double'; minimum is 4.9406564584124654E-324}}

double d2 = 1.7E+309; // expected-warning {{magnitude of floating-point constant too large for type 'double'; maximum is 1.7976931348623157E+308}}

double d3 = 0x0.42p42; // no-warning

double d4 = 0x0.42p-4200; // expected-warning {{magnitude of floating-point constant too small for type 'double'; minimum is 4.9406564584124654E-324}}

double d5 = 0x0.42p+4200; // expected-warning {{magnitude of floating-point constant too large for type 'double'; maximum is 1.7976931348623157E+308}}

long double ld0 = 0.42L; // no-warning

long double ld1 = 3.6E-4952L; // expected-warning {{magnitude of floating-point constant too small for type 'long double'; minimum is 3.64519953188247460253E-4951}}

long double ld2 = 1.2E+4932L; // expected-warning {{magnitude of floating-point constant too large for type 'long double'; maximum is 1.18973149535723176502E+4932}}

long double ld3 = 0x0.42p42L; // no-warning

long double ld4 = 0x0.42p-42000L; // expected-warning {{magnitude of floating-point constant too small for type 'long double'; minimum is 3.64519953188247460253E-4951}}

long double ld5 = 0x0.42p+42000L; // expected-warning {{magnitude of floating-point constant too large for type 'long double'; maximum is 1.18973149535723176502E+4932}}
