// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -verify -fptrauth-intrinsics %s

char __ptrauth(0) a;
// expected-error@-1{{'__ptrauth' qualifier only applies to pointer or pointer sized integer types; 'char' is invalid}}
unsigned char __ptrauth(0) b;
// expected-error@-1{{'__ptrauth' qualifier only applies to pointer or pointer sized integer types; 'unsigned char' is invalid}}
short __ptrauth(0) c;
// expected-error@-1{{'__ptrauth' qualifier only applies to pointer or pointer sized integer types; 'short' is invalid}}
unsigned short __ptrauth(0) d;
// expected-error@-1{{'__ptrauth' qualifier only applies to pointer or pointer sized integer types; 'unsigned short' is invalid}}
int __ptrauth(0) e;
// expected-error@-1{{'__ptrauth' qualifier only applies to pointer or pointer sized integer types; 'int' is invalid}}
unsigned int __ptrauth(0) f;
// expected-error@-1{{'__ptrauth' qualifier only applies to pointer or pointer sized integer types; 'unsigned int' is invalid}}
__int128_t __ptrauth(0) g;
// expected-error@-1{{'__ptrauth' qualifier only applies to pointer or pointer sized integer types; '__int128_t' (aka '__int128') is invalid}}
unsigned short __ptrauth(0) h;
// expected-error@-1{{'__ptrauth' qualifier only applies to pointer or pointer sized integer types; 'unsigned short' is invalid}}

unsigned long long __ptrauth(0) i;
long long __ptrauth(0) j;
__SIZE_TYPE__ __ptrauth(0) k;
const unsigned long long __ptrauth(0) l;
const long long __ptrauth(0) m;
const __SIZE_TYPE__ __ptrauth(0) n;

struct S1 {
  __SIZE_TYPE__ __ptrauth(0) f0;
};

void x(unsigned long long __ptrauth(0) f0);
// expected-error@-1{{parameter type may not be qualified with '__ptrauth'; type is '__ptrauth(0,0,0) unsigned long long'}}

unsigned long long __ptrauth(0) y();
// expected-error@-1{{return type may not be qualified with '__ptrauth'; type is '__ptrauth(0,0,0) unsigned long long'}}
