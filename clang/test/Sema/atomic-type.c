// RUN: %clang_cc1 %s -verify -fsyntax-only
// RUN: %clang_cc1 %s -verify -fsyntax-only -triple x86_64-pc-windows-msvc

// Basic parsing/Sema tests for _Atomic
// No operations are actually supported on objects of this type yet.
// The qualifier syntax is not supported yet.
_Atomic(int) t1;
_Atomic(int) *t2 = &t1;
void testf(void*);
void f(void) {
  _Atomic(_Atomic(int)*) t3;
  _Atomic(_Atomic(int)*) *t4[2] = { &t3, 0 };
  testf(t4);
}
extern _Atomic(int (*)(int(*)[], int(*)[10])) mergetest;
extern _Atomic(int (*)(int(*)[10], int(*)[])) mergetest;
extern _Atomic(int (*)(int(*)[10], int(*)[10])) mergetest;

_Atomic(int(void)) error1; // expected-error {{_Atomic cannot be applied to function type}}
_Atomic(struct ErrorS) error2; // expected-error {{_Atomic cannot be applied to incomplete type}} expected-note {{forward declaration}}
_Atomic(int[10]) error3; // expected-error {{_Atomic cannot be applied to array type}}
_Atomic(const int) error4; // expected-error {{_Atomic cannot be applied to qualified type}}
_Atomic(_Atomic(int)) error5; // expected-error {{_Atomic cannot be applied to atomic type}}

// GH111295: _Atomic(T) is never less aligned than T.
typedef __attribute__((aligned(8))) int alignedint;
_Static_assert(sizeof(alignedint) == sizeof(int), "");
_Static_assert(_Alignof(alignedint) == 8, "");
_Static_assert(sizeof(_Atomic(alignedint)) == sizeof(int), "");
_Static_assert(_Alignof(_Atomic(alignedint)) == 8, "");

typedef __attribute__((aligned(16))) char alignedchar;
_Static_assert(sizeof(_Atomic(alignedchar)) == 1, "");
_Static_assert(_Alignof(_Atomic(alignedchar)) == 16, "");

typedef __attribute__((aligned)) int defaultalignedint;
_Static_assert(_Alignof(_Atomic(defaultalignedint)) == _Alignof(defaultalignedint), "");

enum __attribute__((aligned(8))) alignedenum { AE };
_Static_assert(sizeof(_Atomic(enum alignedenum)) == sizeof(enum alignedenum), "");
_Static_assert(_Alignof(_Atomic(enum alignedenum)) == 8, "");

struct S3 { char c[3]; };
typedef struct S3 __attribute__((aligned(16))) alignedS3;
_Static_assert(sizeof(alignedS3) == 3, "");
_Static_assert(sizeof(_Atomic(alignedS3)) == sizeof(_Atomic(struct S3)), "");
_Static_assert(_Alignof(_Atomic(alignedS3)) == 16, "");

struct Wrap { char c; _Atomic(alignedint) a; };
_Static_assert(__builtin_offsetof(struct Wrap, a) == 8, "");
_Static_assert(sizeof(struct Wrap) == 16, "");
_Static_assert(_Alignof(struct Wrap) == 8, "");

// Other atomic types are still rounded up to a power of two.
_Static_assert(sizeof(_Atomic(char)) == 1 && _Alignof(_Atomic(char)) == 1, "");
_Static_assert(sizeof(_Atomic(int)) == sizeof(int) && _Alignof(_Atomic(int)) == sizeof(int), "");
_Static_assert(sizeof(_Atomic(long long)) == 8 && _Alignof(_Atomic(long long)) == 8, "");
_Static_assert(sizeof(_Atomic(struct S3)) == 4 && _Alignof(_Atomic(struct S3)) == 4, "");
typedef __attribute__((aligned(1))) int underalignedint;
_Static_assert(_Alignof(underalignedint) == 1, "");
_Static_assert(_Alignof(_Atomic(underalignedint)) == _Alignof(_Atomic(int)), "");

// _Atomic(T) keeps T's alignment requirement and resists #pragma pack like T.
typedef __attribute__((aligned(4))) int alignedint4;
#pragma pack(1)
struct PackedPlain { char c; alignedint4 b; alignedint a; };
struct PackedAtomic { char c; _Atomic(alignedint4) b; _Atomic(alignedint) a; };
#pragma pack()
_Static_assert(__builtin_offsetof(struct PackedAtomic, b) == __builtin_offsetof(struct PackedPlain, b), "");
_Static_assert(__builtin_offsetof(struct PackedAtomic, a) == __builtin_offsetof(struct PackedPlain, a), "");
_Static_assert(sizeof(struct PackedAtomic) == sizeof(struct PackedPlain), "");
