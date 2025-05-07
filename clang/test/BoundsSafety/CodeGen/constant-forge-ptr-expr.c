// REQUIRES: x86-registered-target

// RUN: %clang_cc1 %s -O0 -fbounds-safety -triple x86_64 -S -o - | FileCheck --check-prefixes CHECK,CHECKBS %s
// RUN: %clang_cc1 %s -O0 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -triple x86_64 -S -o - | FileCheck --check-prefixes CHECK,CHECKBS %s
// RUN: %clang_cc1 %s -O0 -fexperimental-bounds-safety-attributes -x c -triple x86_64 -S -o - | FileCheck --check-prefixes CHECK %s
// RUN: %clang_cc1 %s -O0 -fexperimental-bounds-safety-attributes -x c++ -triple x86_64 -S -o - | FileCheck --check-prefixes CHECK %s
// RUN: %clang_cc1 %s -O0 -fexperimental-bounds-safety-attributes -x objective-c -triple x86_64 -S -o - | FileCheck --check-prefixes CHECK %s
// RUN: %clang_cc1 %s -O0 -fexperimental-bounds-safety-attributes -x objective-c++ -triple x86_64 -S -o - | FileCheck --check-prefixes CHECK %s

#include <ptrcheck.h>

/* Define to nothing in attribute-only mode */
#ifndef __bidi_indexable
#define __bidi_indexable
#endif

struct Foo {
  void *dummy;
  int buf[18];
};
_Static_assert(sizeof(struct Foo) == 80, "sizeof(struct Foo) should be 80");

int arr[2];
int pad[2];
struct Foo foo[8];

int *ptrSingle = __unsafe_forge_single(int *, 0);
// CHECK-LABEL: ptrSingle:
// CHECK:   .quad   0
// CHECK:   .size   ptrSingle, 8

/* XXX: won't work until __unsafe_forge_single knows the size it's forging
int *__indexable ptrIndexable = __unsafe_forge_single(int *, 0);
// TODO-CHECK-LABEL: ptrIndexable:
// TODO-CHECK:   .zero   16
// TODO-CHECK:   .size   ptrIndexable, 16
 */

int *__bidi_indexable ptrBidiIndexable =
    __unsafe_forge_bidi_indexable(int *, arr + 2, 14);
// CHECKBS-LABEL: ptrBidiIndexable:
// CHECKBS:   .quad   arr+8
// CHECKBS:   .quad   arr+8+14
// CHECKBS:   .quad   arr+8

int *__bidi_indexable ptrBidiIndexable2 = __unsafe_forge_single(int *, &arr[0]);
// CHECKBS-LABEL: ptrBidiIndexable2:
// CHECKBS:   .quad   arr
// CHECKBS:   .quad   arr+4
// CHECKBS:   .quad   arr

int *__bidi_indexable ptrBidiIndexable3 = __unsafe_forge_single(int *, 0x1234);
// CHECKBS-LABEL: ptrBidiIndexable3:
// CHECKBS:   .quad   4660
// CHECKBS:   .quad   4664
// CHECKBS:   .quad   4660

int *__bidi_indexable ptrBidiIndexable4 = __unsafe_forge_bidi_indexable(int *, 8000, 16);
// CHECKBS-LABEL: ptrBidiIndexable4:
// CHECKBS:   .quad   8000
// CHECKBS:   .quad   8016
// CHECKBS:   .quad   8000

// (4 * 80) + 8 + 4 * 9 = 364
int *__bidi_indexable ptrBidiIndexable5 = __unsafe_forge_bidi_indexable(int *, &foo[4].buf[9], 16);
// CHECKBS-LABEL: ptrBidiIndexable5:
// CHECKBS:   .quad   foo+364
// CHECKBS:   .quad   foo+364+16
// CHECKBS:   .quad   foo+364
// CHECKBS:   .size   ptrBidiIndexable5, 24

#define FOO_PLUS_364 (&foo[4].buf[9])
#define FOO_PLUS_364_LEN_16 __unsafe_forge_bidi_indexable(int *, FOO_PLUS_364, 16)
#define FOO_PLUS_380_LEN_0 (FOO_PLUS_364_LEN_16 + 4)
int *__bidi_indexable ptrBidiIndexable6 =
  __unsafe_forge_bidi_indexable(int *, FOO_PLUS_380_LEN_0, 8) + 4;
// CHECKBS-LABEL: ptrBidiIndexable6:
// CHECKBS:   .quad   foo+380+16
// CHECKBS:   .quad   foo+380+8
// CHECKBS:   .quad   foo+380
// CHECKBS:   .size   ptrBidiIndexable6, 24

int *ptrSingle2 = __unsafe_forge_bidi_indexable(int *, 0, 0);
// CHECKBS-LABEL: ptrSingle2:
// CHECKBS:   .quad   0
// CHECKBS:   .size   ptrSingle2, 8

int *ptrSingle3 = __unsafe_forge_bidi_indexable(int *, arr + 2, 4);
// CHECKBS-LABEL: ptrSingle3:
// CHECKBS:   .quad   arr+8
// CHECKBS:   .size   ptrSingle3, 8

/* This is invalid; converting a pointer with a size of 3 to an int *__single
   never works. */
// int *ptrSingle4 = __unsafe_forge_bidi_indexable(int *, arr + 2, 3);

int *ptrSingle5 = __unsafe_forge_single(int *, __unsafe_forge_bidi_indexable(int *, arr + 2, 3) + 4);
// CHECKBS-LABEL: ptrSingle5:
// CHECKBS:   .quad   arr+24
// CHECKBS:   .size   ptrSingle5, 8

int *ptrSingle6 = __unsafe_forge_single(int *, &arr[0]);
// CHECK-LABEL: ptrSingle6:
// CHECK:   .quad   arr
// CHECK:   .size   ptrSingle6, 8

int *ptrSingle7 = __unsafe_forge_single(int *, 1337);
// CHECK-LABEL: ptrSingle7:
// CHECK:   .quad   1337
// CHECK:   .size   ptrSingle7, 8

int *ptrSingle8 = __unsafe_forge_single(int *, __unsafe_forge_single(int *, arr + 2));
// CHECK-LABEL: ptrSingle8:
// CHECK:   .quad   arr+8
// CHECK:   .size   ptrSingle8, 8

int *__terminated_by(5) ptrTerminated = __unsafe_forge_terminated_by(int *, arr, 5);
// CHECK-LABEL: ptrTerminated:
// CHECK:   .quad   arr
// CHECK:   .size   ptrTerminated, 8

char *__null_terminated ptrNt1 = __unsafe_forge_null_terminated(char *, __unsafe_forge_bidi_indexable(char *, arr+12, 10));
// CHECKBS-LABEL: ptrNt1:
// CHECKBS:   .quad   arr+48
// CHECKBS:   .size   ptrNt1, 8

char *__null_terminated ptrNt2 = __unsafe_forge_null_terminated(char *, __unsafe_forge_single(char *, arr+2));
// CHECK-LABEL: ptrNt2:
// CHECK:   .quad   arr+8
// CHECK:   .size   ptrNt2, 8
