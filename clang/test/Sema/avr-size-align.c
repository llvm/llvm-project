// RUN: %clang_cc1 %s -triple avr -fsyntax-only

_Static_assert(sizeof(char) == 1, "sizeof(char) == 1");
_Static_assert(_Alignof(char) == 1, "_Alignof(char) == 1");
_Static_assert(__alignof(char) == 1, "__alignof(char) == 1");

_Static_assert(sizeof(short) == 2, "sizeof(short) == 2");
_Static_assert(_Alignof(short) == 1, "_Alignof(short) == 1");
_Static_assert(__alignof(short) == 1, "__alignof(short) == 1");

_Static_assert(sizeof(unsigned short) == 2, "sizeof(unsigned short) == 2");
_Static_assert(_Alignof(unsigned short) == 1, "_Alignof(unsigned short) == 1");
_Static_assert(__alignof(unsigned short) == 1, "__alignof(unsigned short) == 1");

_Static_assert(sizeof(int) == 2, "sizeof(int) == 2");
_Static_assert(_Alignof(int) == 1, "_Alignof(int) == 1");
_Static_assert(__alignof(int) == 1, "__alignof(int) == 1");

_Static_assert(sizeof(unsigned int) == 2, "sizeof(unsigned int) == 2");
_Static_assert(_Alignof(unsigned int) == 1, "_Alignof(unsigned int) == 1");
_Static_assert(__alignof(unsigned int) == 1, "__alignof(unsigned int) == 1");

_Static_assert(sizeof(long) == 4, "sizeof(long) == 4");
_Static_assert(_Alignof(long) == 1, "_Alignof(long) == 1");
_Static_assert(__alignof(long) == 1, "__alignof(long) == 1");

_Static_assert(sizeof(unsigned long) == 4, "sizeof(unsigned long) == 4");
_Static_assert(_Alignof(unsigned long) == 1, "_Alignof(unsigned long) == 1");
_Static_assert(__alignof(unsigned long) == 1, "__alignof(unsigned long) == 1");

_Static_assert(sizeof(long long) == 8, "sizeof(long long) == 8");
_Static_assert(_Alignof(long long) == 1, "_Alignof(long long) == 1");
_Static_assert(__alignof(long long) == 1, "__alignof(long long) == 1");

_Static_assert(sizeof(unsigned long long) == 8, "sizeof(unsigned long long) == 8");
_Static_assert(_Alignof(unsigned long long) == 1, "_Alignof(unsigned long long) == 1");
_Static_assert(__alignof(unsigned long long) == 1, "__alignof(unsigned long long) == 1");

_Static_assert(sizeof(float) == 4, "sizeof(float) == 4");
_Static_assert(_Alignof(float) == 1, "_Alignof(float) == 1");
_Static_assert(__alignof(float) == 1, "__alignof(float) == 1");

_Static_assert(sizeof(double) == 4, "sizeof(double) == 4");
_Static_assert(_Alignof(double) == 1, "_Alignof(double) == 1");
_Static_assert(__alignof(double) == 1, "__alignof(double) == 1");

_Static_assert(sizeof(long double) == 4, "sizeof(long double) == 4");
_Static_assert(_Alignof(long double) == 1, "_Alignof(long double) == 1");
_Static_assert(__alignof(long double) == 1, "__alignof(long double) == 1");
