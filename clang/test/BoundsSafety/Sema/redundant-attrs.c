
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>
#include "redundant-attrs.h"

int *__bidi_indexable __bidi_indexable ptrBoundBound; // expected-warning{{pointer annotated with __bidi_indexable multiple times. Annotate only once to remove this warning}}
int *__indexable __bidi_indexable ptrArrayBound; // expected-error{{pointer cannot have more than one bound attribute}}
int *__indexable __single ptrArrayThin; // expected-error{{pointer cannot have more than one bound attribute}}
int *__bidi_indexable *__indexable ptrBoundPtrArray;
int *__bidi_indexable *__bidi_indexable ptrBoundPtrBound;
int *__indexable __single *ptrArrayThinCast; // expected-error{{pointer cannot have more than one bound attribute}}
int *__bidi_indexable __single *ptrBoudCastThin; // expected-error{{pointer cannot have more than one bound attribute}}

// Using `#pragma clang abi_ptr_attr set(X Y)` must fail if X and Y are different
// attributes, but you must be able to use `#pragma clang abi_ptr_attr set(X)` and
// then `#pragma clang abi_ptr_attr set(Y)` (which the macros below expand to).
__ptrcheck_abi_assume_unsafe_indexable()
__ptrcheck_abi_assume_single() // Only default, explicit bounds override instead of conflict.

int *__unsafe_indexable foo(int * __unsafe_indexable x) {
  int * __unsafe_indexable * __unsafe_indexable __unsafe_indexable y = &x; // expected-warning{{pointer annotated with __unsafe_indexable multiple times. Annotate only once to remove this warning}}
  return *y;
}

// expected-error@+1{{pointer cannot have more than one bound attribute}}
_Pragma("clang abi_ptr_attr set(single unsafe_indexable)")
void bar(int * x);

typedef int *__bidi_indexable bidiPtr;
typedef int *__bidi_indexable __bidi_indexable bidiBidiPtr; // expected-warning{{pointer annotated with __bidi_indexable multiple times. Annotate only once to remove this warning}}
bidiPtr __bidi_indexable ptrBoundBound2;

#define bidiPtr2 int *__bidi_indexable
bidiPtr2 __bidi_indexable ptrBoundBound3;
bidiPtr2_header __bidi_indexable ptrBoundBound3_header;

typedef int *intPtr;
intPtr __single __single ptrBoundBound4; // expected-warning{{pointer annotated with __single multiple times. Annotate only once to remove this warning}}
char * __null_terminated __null_terminated ptrBoundBound5; // expected-warning{{pointer annotated with __terminated_by multiple times. Annotate only once to remove this warning}}

int len = 0;
int * __counted_by(len) __counted_by(len) ptrBoundBound6; // expected-error{{pointer cannot have more than one count attribute}}

struct S1 {
    int size;
    int arrBoundBound[__counted_by(size) __counted_by(size)]; // expected-error{{array cannot have more than one count attribute}}
};
struct S2 {
    int size;
    int len;
    int arrBoundBound[__counted_by(size) __counted_by(len)]; // expected-error{{array cannot have more than one count attribute}}
};

#define intPtr2 int *
intPtr2 __single __single ptrBoundBound7; // expected-warning{{pointer annotated with __single multiple times. Annotate only once to remove this warning}}
intPtr2_header __single __single ptrBoundBound7_header; // expected-warning{{pointer annotated with __single multiple times. Annotate only once to remove this warning}}
int * __counted_by(len) __counted_by(len) ptrBoundBound8; // expected-error{{pointer cannot have more than one count attribute}}

bidiPtr __indexable ptrBoundBoundMismatch; // expected-error{{pointer cannot have more than one bound attribute}}
bidiPtr2 __indexable ptrBoundBoundMismatch2; // expected-error{{pointer cannot have more than one bound attribute}}
bidiPtr2_header __indexable ptrBoundBoundMismatch3; // expected-error{{pointer cannot have more than one bound attribute}}

#define BIDI_INDEXABLE(T, X)    do { T __bidi_indexable X; } while (0)
void macro_decl(void) {
    BIDI_INDEXABLE(int * __bidi_indexable, varName);
}

#define BIDI_INDEXABLE2(X)    do { int * __bidi_indexable X; } while (0)
void macro_decl2(void) {
    BIDI_INDEXABLE2(__bidi_indexable varName);
}

void macro_decl3(void) {
    BIDI_INDEXABLE(int * __bidi_indexable __bidi_indexable, varName); // expected-warning{{pointer annotated with __bidi_indexable multiple times. Annotate only once to remove this warning}}
    BIDI_INDEXABLE_header(int * __bidi_indexable __bidi_indexable, varName_header); // expected-warning{{pointer annotated with __bidi_indexable multiple times. Annotate only once to remove this warning}}
}

void macro_decl4(void) {
    BIDI_INDEXABLE2(__bidi_indexable __bidi_indexable varName); // expected-warning{{pointer annotated with __bidi_indexable multiple times. Annotate only once to remove this warning}}
    BIDI_INDEXABLE2_header(__bidi_indexable __bidi_indexable varName_header); // expected-warning{{pointer annotated with __bidi_indexable multiple times. Annotate only once to remove this warning}}
}

int * __single * __single __single ptrBoundBoundNested; // expected-warning{{pointer annotated with __single multiple times. Annotate only once to remove this warning}}
int * __single __single * __single ptrBoundBoundNested2; // expected-warning{{pointer annotated with __single multiple times. Annotate only once to remove this warning}}
_Atomic(int * __single) __single * __single ptrBoundBoundNestedAtomic;
_Atomic(int * __single) __single __single * __single ptrBoundBoundNestedAtomic2; // expected-warning{{pointer annotated with __single multiple times. Annotate only once to remove this warning}}
_Atomic(int * __single __single) __single * __single ptrBoundBoundNestedAtomic3; // expected-warning{{pointer annotated with __single multiple times. Annotate only once to remove this warning}}

int * __null_terminated * __null_terminated __null_terminated ptrBoundBoundNestedNullTerm; // expected-warning{{pointer annotated with __terminated_by multiple times. Annotate only once to remove this warning}}
int * __null_terminated __null_terminated * __null_terminated ptrBoundBoundNestedNullTerm2; // expected-warning{{pointer annotated with __terminated_by multiple times. Annotate only once to remove this warning}}

typedef int *__null_terminated nullTermPtr;
nullTermPtr __null_terminated * __null_terminated ptrBoundBoundNestedNullTermTypedef;
nullTermPtr __null_terminated __null_terminated * __null_terminated ptrBoundBoundNestedNullTermTypedef2; // expected-warning{{pointer annotated with __terminated_by multiple times. Annotate only once to remove this warning}}
nullTermPtr __terminated_by(1)  ptrTerm0Term1; // expected-error{{pointer cannot have more than one terminator attribute}}
                                               // expected-note@-1{{conflicting arguments for terminator were '0' and '1'}}

typedef int *__null_terminated __null_terminated nullTermNullTermPtr; // expected-warning{{pointer annotated with __terminated_by multiple times. Annotate only once to remove this warning}}
nullTermNullTermPtr * __null_terminated ptrBoundBoundNestedNullTermTypedef3;

#define nullTermPtr2 int *__null_terminated
nullTermPtr2 __null_terminated ptrNullTermNullTerm;
nullTermPtr2_header __null_terminated ptrNullTermNullTerm_header;

int * __attribute__((aligned (16))) __single ptrUnrelatedAttributeBound;
int * __null_terminated __attribute__((aligned (16))) __null_terminated ptrUnrelatedAttributeBoundBound; // expected-warning{{pointer annotated with __terminated_by multiple times. Annotate only once to remove this warning}}
nullTermPtr __attribute__((aligned (16))) __null_terminated ptrUnrelatedAttributeBoundBound2;

int * __attribute__((__bidi_indexable__)) __attribute__((__bidi_indexable__)) ptrBoundBoundNoMacro; // expected-warning{{pointer annotated with __bidi_indexable multiple times. Annotate only once to remove this warning}}
#define nullTermPtr3 int * __attribute__((__bidi_indexable__))
nullTermPtr3 __attribute__((__bidi_indexable__)) ptrBoundBoundNoMacro2;
nullTermPtr3_header __attribute__((__bidi_indexable__)) ptrBoundBoundNoMacro3;

int *__terminated_by(0) __terminated_by(0u) ptrTerminatedByDifferentSignedness; //expected-warning{{pointer annotated with __terminated_by multiple times. Annotate only once to remove this warning}}
int *__terminated_by(0) __terminated_by(0l) ptrTerminatedByDifferentWidth; //expected-warning{{pointer annotated with __terminated_by multiple times. Annotate only once to remove this warning}}
int *const __terminated_by(0) _Nullable __terminated_by(0) * _Nullable __terminated_by(0) ptrTerminatedNestedAndNullable; //expected-warning{{pointer annotated with __terminated_by multiple times. Annotate only once to remove this warning}}

typedef int *_Nullable __bidi_indexable nullableBidiPtr;
nullableBidiPtr __single ptrNullableBidiSingleConflict; //expected-error{{pointer cannot have more than one bound attribute}}

typedef int *_Nullable nullablePtr;
nullablePtr __single ptrAttributedTypeSeparateDecl;

typedef int *_Nullable __null_terminated __bidi_indexable nullableNTBidiPtr; //expected-error{{'__terminated_by' attribute currently can be applied only to '__single' pointers}}
nullableNTBidiPtr ptrNullableNTBidi;
