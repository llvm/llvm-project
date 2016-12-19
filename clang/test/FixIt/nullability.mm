// RUN: %clang_cc1 -fsyntax-only -fblocks -std=gnu++11 -I %S/Inputs -verify %s
// RUN: not %clang_cc1 -fdiagnostics-parseable-fixits -fblocks -std=gnu++11 -I %S/Inputs %s >%t.txt 2>&1
// RUN: FileCheck %s < %t.txt
// RUN: FileCheck %S/Inputs/nullability.h < %t.txt

#include "nullability.h"

#pragma clang assume_nonnull begin

extern void *array[2]; // expected-warning {{inferring '_Nonnull' for pointer type within array is deprecated}}
// CHECK: fix-it:"{{.*}}nullability.mm":{[[@LINE-1]]:14-[[@LINE-1]]:14}:" _Nonnull "

extern void* array2[2]; // expected-warning {{inferring '_Nonnull' for pointer type within array is deprecated}}
// CHECK: fix-it:"{{.*}}nullability.mm":{[[@LINE-1]]:13-[[@LINE-1]]:13}:" _Nonnull"

extern void *nestedArray[2][3]; // expected-warning {{inferring '_Nonnull' for pointer type within array is deprecated}}
// CHECK: fix-it:"{{.*}}nullability.mm":{[[@LINE-1]]:14-[[@LINE-1]]:14}:" _Nonnull "

// No fix-its on either the macro definition or instantiation.
// CHECK-NOT: fix-it:"{{.*}}nullability.mm":{[[@LINE+2]]
// CHECK-NOT: fix-it:"{{.*}}nullability.mm":{[[@LINE+2]]
#define PTR(X) X *
extern PTR(void) array[2]; // expected-warning {{inferring '_Nonnull' for pointer type within array is deprecated}}


typedef const void *CFTypeRef;

extern CFTypeRef typedefArray[2]; // expected-warning {{inferring '_Nonnull' for pointer type within array is deprecated}}
// CHECK: fix-it:"{{.*}}nullability.mm":{[[@LINE-1]]:17-[[@LINE-1]]:17}:" _Nonnull"


extern void *&ref; // expected-warning {{inferring '_Nonnull' for pointer type within reference is deprecated}}
// CHECK: fix-it:"{{.*}}nullability.mm":{[[@LINE-1]]:14-[[@LINE-1]]:14}:"_Nonnull"

extern void * &ref2; // expected-warning {{inferring '_Nonnull' for pointer type within reference is deprecated}}
// CHECK: fix-it:"{{.*}}nullability.mm":{[[@LINE-1]]:14-[[@LINE-1]]:14}:" _Nonnull"

extern void *&&ref3; // expected-warning {{inferring '_Nonnull' for pointer type within reference is deprecated}}
// CHECK: fix-it:"{{.*}}nullability.mm":{[[@LINE-1]]:14-[[@LINE-1]]:14}:"_Nonnull"

extern void * &&ref4; // expected-warning {{inferring '_Nonnull' for pointer type within reference is deprecated}}
// CHECK: fix-it:"{{.*}}nullability.mm":{[[@LINE-1]]:14-[[@LINE-1]]:14}:" _Nonnull"

extern void *(&arrayRef)[2]; // expected-warning {{inferring '_Nonnull' for pointer type within array is deprecated}}
// CHECK: fix-it:"{{.*}}nullability.mm":{[[@LINE-1]]:14-[[@LINE-1]]:14}:"_Nonnull"

extern void * (&arrayRef2)[2]; // expected-warning {{inferring '_Nonnull' for pointer type within array is deprecated}}
// CHECK: fix-it:"{{.*}}nullability.mm":{[[@LINE-1]]:14-[[@LINE-1]]:14}:" _Nonnull"

extern CFTypeRef &typedefRef; // expected-warning {{inferring '_Nonnull' for pointer type within reference is deprecated}}
// CHECK: fix-it:"{{.*}}nullability.mm":{[[@LINE-1]]:17-[[@LINE-1]]:17}:" _Nonnull"
extern CFTypeRef& typedefRef2; // expected-warning {{inferring '_Nonnull' for pointer type within reference is deprecated}}
// CHECK: fix-it:"{{.*}}nullability.mm":{[[@LINE-1]]:17-[[@LINE-1]]:17}:" _Nonnull "


void arrayNameless(void *[]); // expected-warning {{inferring '_Nonnull' for pointer type within array is deprecated}}
// CHECK: fix-it:"{{.*}}nullability.mm":{[[@LINE-1]]:26-[[@LINE-1]]:26}:"_Nonnull"

void arrayNameless2(void * []); // expected-warning {{inferring '_Nonnull' for pointer type within array is deprecated}}
// CHECK: fix-it:"{{.*}}nullability.mm":{[[@LINE-1]]:27-[[@LINE-1]]:27}:" _Nonnull"

void arrayNamelessTypedef(CFTypeRef[]); // expected-warning {{inferring '_Nonnull' for pointer type within array is deprecated}}
// CHECK: fix-it:"{{.*}}nullability.mm":{[[@LINE-1]]:36-[[@LINE-1]]:36}:" _Nonnull "

void arrayNamelessTypedef2(CFTypeRef []); // expected-warning {{inferring '_Nonnull' for pointer type within array is deprecated}}
// CHECK: fix-it:"{{.*}}nullability.mm":{[[@LINE-1]]:37-[[@LINE-1]]:37}:" _Nonnull"


extern int (*pointerToArray)[2]; // no-warning
int checkTypePTA = pointerToArray; // expected-error {{cannot initialize a variable of type 'int' with an lvalue of type 'int (* _Nonnull)[2]'}}

int **arrayOfNestedPointers[2]; // no-warning
int checkTypeANP = arrayOfNestedPointers; // expected-error {{cannot initialize a variable of type 'int' with an lvalue of type 'int **[2]'}}

CFTypeRef *arrayOfNestedPointersTypedef[2]; // no-warning
int checkTypeANPT = arrayOfNestedPointersTypedef; // expected-error {{cannot initialize a variable of type 'int' with an lvalue of type 'CFTypeRef *[2]'}}

#pragma clang assume_nonnull end
