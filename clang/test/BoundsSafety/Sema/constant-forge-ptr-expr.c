
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

int arr[4];
int len = 4;
int *__bidi_indexable ptrBidi = &((int*)__unsafe_forge_bidi_indexable(int *, 0, 16))[10];
int *__bidi_indexable ptrBidi2 = __unsafe_forge_bidi_indexable(int *, arr + 2, 2);
int *__bidi_indexable ptrBidi3 = __unsafe_forge_bidi_indexable(int *, 8000, len); // expected-error{{initializer element is not a compile-time constant}}
int *__indexable ptrArr = &((int*)__unsafe_forge_bidi_indexable(int *, 0, 10))[-1]; // expected-error{{initializer element is not a compile-time constant}}
int *__indexable ptrArr2 = &((int*)__unsafe_forge_bidi_indexable(int *, 0, 10))[1];
int *__indexable ptrArr3 = &((int*)__unsafe_forge_bidi_indexable(int *, 0, 10))[2]; // expected-error{{initializer element is not a compile-time constant}}
// XXX: can't work until __unsafe_forge_single has a size
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbounds-attributes-implicit-conversion-single-to-explicit-indexable"
int *__indexable ptrArr4 = __unsafe_forge_single(int *, 10); // expected-error{{initializer element is not a compile-time constant}}
#pragma clang diagnostic pop
int *__single ptrSingle = __unsafe_forge_bidi_indexable(int *, arr, 1); // expected-error{{initializer element is not a compile-time constant}}
int *__single ptrSingle2 = __unsafe_forge_bidi_indexable(int *, 0, 0);
int *__single ptrSingle12 = __unsafe_forge_bidi_indexable(int *, 0, 4);
int *__single ptrSingle3 = __unsafe_forge_bidi_indexable(int *, 2, 4) + 4; // expected-error{{initializer element is not a compile-time constant}}
int *__single ptrSingle4 = __unsafe_forge_bidi_indexable(int *, 2, 5) + 4; // expected-error{{initializer element is not a compile-time constant}}
int *__single ptrSingle5 = __unsafe_forge_bidi_indexable(int *, 2, 7) + 4; // expected-error{{initializer element is not a compile-time constant}}
int *__single ptrSingle11 = __unsafe_forge_bidi_indexable(int *, 2, 8) + 4; // expected-error{{initializer element is not a compile-time constant}}
int *__single ptrSingle6 = __unsafe_forge_single(int *, 0);
int *__single ptrSingle7 = &((int*)__unsafe_forge_bidi_indexable(int *, arr, 16))[1];
int *__single ptrSingle8 = &((int*)__unsafe_forge_bidi_indexable(int *, arr, 16))[-2]; // expected-error{{initializer element is not a compile-time constant}}
int *__single ptrSingle9 = __unsafe_forge_bidi_indexable(int *, &arr[4], 8) + 4; // expected-error{{initializer element is not a compile-time constant}}
int *__single ptrSingle10 = __unsafe_forge_bidi_indexable(int *, &arr[4], 8) + 8; // expected-error{{initializer element is not a compile-time constant}}
int *__single ptrSingle13 = __unsafe_forge_single(int *, arr);
int *__terminated_by(5) ptrTerminated = __unsafe_forge_terminated_by(int *, arr, 5);
int *__terminated_by(5) ptrTerminated2 = __unsafe_forge_terminated_by(int *, arr, 6); // expected-error{{pointers with incompatible terminators initializing 'int *__single __terminated_by(5)' (aka 'int *__single') with an expression of incompatible type 'int *__single __terminated_by(6)' (aka 'int *__single')}}
int *__terminated_by(5) ptrTerminated3 = __unsafe_forge_terminated_by(int *, 7, 5);
// The attribute on cast is missing macro identifer info. 
// expected-error@+2{{'__terminated_by__' attribute requires an integer constant}}
// expected-error@+1{{initializing 'int *__single __terminated_by(4)' (aka 'int *__single') with an expression of incompatible type 'int *__single' is an unsafe operation; use '__unsafe_terminated_by_from_indexable()' or '__unsafe_forge_terminated_by()' to perform this conversion}}
int *__terminated_by(4) ptrTerminated4 = __unsafe_forge_terminated_by(int *, arr, len);
int *__null_terminated ptrNullTerminated = __unsafe_forge_null_terminated(int *, arr);
// expected-error@+2{{'__terminated_by__' attribute requires an integer constant}}
// expected-error@+1{{initializing 'int *__single __terminated_by(0)' (aka 'int *__single') with an expression of incompatible type 'int *__single' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
int *__terminated_by(0) ptrTerminated5 = __unsafe_forge_terminated_by(int *, 0, arr);
int *__terminated_by(0) ptrTerminated6 = __unsafe_forge_terminated_by(int *, 0, 0);
// expected-error@+3{{initializer element is not a compile-time constant}}
// expected-error@+2{{'__terminated_by__' attribute requires an integer constant}}
// expected-error@+1{{initializing 'int *__single __terminated_by(0)' (aka 'int *__single') with an expression of incompatible type 'int *__single' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
int *__terminated_by(0) ptrTerminated7 = __unsafe_forge_terminated_by(int *, 1, arr);

// rdar://84175702
char *c = __unsafe_forge_bidi_indexable(char *, "a", 3); // expected-error{{initializer element is not a compile-time constant}}

// expected-error@+3{{initializing 'char *__single __terminated_by(0)' (aka 'char *__single') with an expression of incompatible type 'char *__bidi_indexable' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
// expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
// expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
char *__null_terminated ptrNt1 = __unsafe_forge_bidi_indexable(char *, arr, 10);

// expected-error@+3{{initializing 'char *__single __terminated_by(0)' (aka 'char *__single') with an expression of incompatible type 'char *__bidi_indexable' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
// expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
// expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
char *__null_terminated ptrNt2 = __unsafe_forge_bidi_indexable(char *, arr+12, 10);

// expected-error@+2{{initializer element is not a compile-time constant}}
// expected-error@+1{{initializing 'char *__single __terminated_by(0)' (aka 'char *__single') with an expression of incompatible type 'char *__single' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
char *__null_terminated ptrNt3 = __unsafe_forge_single(char *, arr+12);

// expected-error@+1{{initializer element is not a compile-time constant}}
char *__null_terminated ptrNt4 = __unsafe_null_terminated_from_indexable(__unsafe_forge_bidi_indexable(char *, arr+12, 10));

char *__null_terminated ptrNt5 = __unsafe_forge_null_terminated(char *, __unsafe_forge_bidi_indexable(char *, arr+12, 10));

char *__null_terminated ptrNt6 = __unsafe_forge_null_terminated(char *, __unsafe_forge_single(char *, arr+2));
