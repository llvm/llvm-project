// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -fsyntax-only -verify %s

typedef int __attribute__((stdcall)) Callback(int);
int copied(int) __attribute__((copy((Callback *)0)));
_Static_assert(__builtin_types_compatible_p(__typeof__(copied), Callback),
               "copied calling convention");

typedef int __attribute__((regparm(2))) Registers(int, int);
int registers(int, int) __attribute__((copy((Registers *)0)));
_Static_assert(__builtin_types_compatible_p(__typeof__(registers), Registers),
               "copied register arguments");

typedef int Plain(int);
typedef Plain CopiedType __attribute__((copy((Callback *)0)));
_Static_assert(__builtin_types_compatible_p(CopiedType, Callback),
               "calling convention on a typedef");

Callback *pointer_source;
int (*pointer_copy)(int) __attribute__((copy(pointer_source)));
_Static_assert(__builtin_types_compatible_p(__typeof__(pointer_copy), Callback *),
               "calling convention on a function pointer");

int __attribute__((fastcall, copy((Registers *)0))) conflict(int, int); // expected-error {{fastcall and regparm attributes are not compatible}}
int variadic(int, ...) __attribute__((copy((Callback *)0))); // expected-warning {{stdcall calling convention is not supported on variadic function}}
