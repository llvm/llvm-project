

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=expected,bs %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify=expected,bs %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -verify=expected,bsa %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c++ -verify=expected,bsa %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c -verify=expected,bsa %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c++ -verify=expected,bsa %s

#include <ptrcheck.h>

/* Define to nothing in attribute-only mode */
#ifndef __indexable
#define __indexable
#endif

#define forge_bidi_indexable(A, B) __unsafe_forge_bidi_indexable(int *, A, B)
#define forge_single(A) __unsafe_forge_single(int *, A)
#define forge_terminated_by(A, B) __unsafe_forge_terminated_by(int *, A, B)

void Test1() {
    int *__indexable ptrArray = forge_bidi_indexable(0, sizeof(int));
    int *__single ptr = forge_single(0);
    int *__terminated_by(0) ptr2 = forge_terminated_by(0, 0);
}

#define MACRO_ZERO 0
#define MACRO_ONE 1
#define MACRO_MINUS_ONE -1

/* In C++ enum is not an int, use macros to make the test pass */
#ifdef __cplusplus
#define ZERO 0
#define ONE 1
#else
enum Enum { ZERO, ONE};
#endif

int global_int_var;
int* global_int_ptr_var;
struct StructFoo {
    int member_int_var;
    int* member_int_ptr_var;
};
int get_negative_int(void) { return -1; }
#define NULL 0

#define MACRO_UINT64_MAX (~0ull)
#define MACRO_UINT32_MAX (~0u)

void Test2() {
    // bs-error@+2{{'__unsafe_forge_bidi_indexable' requires a pointer, array or integer address argument}}
    // bsa-error-re@+1{{{{.*}}'double'{{.*}}pointer type}}
    (void) forge_bidi_indexable(0., 0);
    (void) forge_single(0.); // expected-error{{'__unsafe_forge_single' requires a pointer, array or integer address argument}}
    (void) forge_terminated_by(0., 0); // expected-error{{'__unsafe_forge_terminated_by' requires a pointer, array or integer address argument}}

    (void) forge_bidi_indexable(ZERO, 0);
    (void) forge_bidi_indexable(ONE, 0);
    (void) forge_bidi_indexable(MACRO_ZERO, 0);
    (void) forge_bidi_indexable(MACRO_ONE, 0);
    (void) forge_bidi_indexable(NULL, 0);

    (void) forge_single(ZERO);
    (void) forge_single(ONE);
    (void) forge_single(MACRO_ZERO);
    (void) forge_single(MACRO_ONE);
    (void) forge_single(NULL);

    (void) forge_terminated_by(ZERO, 0);
    (void) forge_terminated_by(ONE, 0);
    (void) forge_terminated_by(MACRO_ZERO, 0);
    (void) forge_terminated_by(MACRO_ONE, 0);
    (void) forge_terminated_by(NULL, 0);

    struct StructFoo F;
    // bs-error@+2{{'__unsafe_forge_bidi_indexable' requires a pointer, array or integer address argument}}
    // bsa-error-re@+1{{{{.*}}'struct StructFoo'{{.*}}pointer type}}
    (void) forge_bidi_indexable(F, 0);
    (void) forge_bidi_indexable(&F, 0);
    (void) forge_bidi_indexable(&F.member_int_var, 0);
    (void) forge_bidi_indexable(F.member_int_ptr_var, 0);

    (void) forge_single(F); // expected-error{{'__unsafe_forge_single' requires a pointer, array or integer address argument}}
    (void) forge_single(&F);
    (void) forge_single(&F.member_int_var);
    (void) forge_single(F.member_int_ptr_var);

    (void) forge_terminated_by(F, 0); // expected-error{{'__unsafe_forge_terminated_by' requires a pointer, array or integer address argument}}
    (void) forge_terminated_by(&F, 0);
    (void) forge_terminated_by(&F.member_int_var, 0);
    (void) forge_terminated_by(F.member_int_ptr_var, 0);

    int local_int_var = 42;
    int* local_int_ptr_var;
    (void) forge_bidi_indexable(&local_int_var, 0);
    (void) forge_bidi_indexable(local_int_ptr_var, 0);
    (void) forge_bidi_indexable(&global_int_var, 0);
    (void) forge_bidi_indexable(global_int_ptr_var, 0);
    (void) forge_bidi_indexable(~0ull, 0);
    (void) forge_bidi_indexable(-1, 0); // bs-error{{negative address argument to '__unsafe_forge_bidi_indexable'}}
    (void) forge_bidi_indexable(MACRO_MINUS_ONE, 0); // bs-error{{negative address argument to '__unsafe_forge_bidi_indexable'}}
    (void) forge_bidi_indexable(&local_int_var, MACRO_UINT64_MAX);
    // bs-error@-1{{negative size argument to '__unsafe_forge_bidi_indexable'}}
    (void) forge_bidi_indexable(0, MACRO_UINT32_MAX);
    (void) forge_bidi_indexable(0, (unsigned char) 240);

    (void) forge_single(&local_int_var);
    (void) forge_single(local_int_ptr_var);
    (void) forge_single(&global_int_var);
    (void) forge_single(global_int_ptr_var);
    (void) forge_single(~0ull);
    (void) forge_single(-1); // expected-error{{negative address argument to '__unsafe_forge_single'}}
    (void) forge_single(MACRO_MINUS_ONE); // expected-error{{negative address argument to '__unsafe_forge_single'}}

    (void) forge_terminated_by(&local_int_var, 0);
    (void) forge_terminated_by(local_int_ptr_var, 0);
    (void) forge_terminated_by(&global_int_var, 0);
    (void) forge_terminated_by(global_int_ptr_var, 0);
    (void) forge_terminated_by(~0ull, 0);
    (void) forge_terminated_by(-1, 0); // expected-error{{negative address argument to '__unsafe_forge_terminated_by'}}
    (void) forge_terminated_by(MACRO_MINUS_ONE, 0); // expected-error{{negative address argument to '__unsafe_forge_terminated_by'}}
}

void Test3() {
    float f;
    (void) forge_bidi_indexable(0, f); // bs-error{{'__unsafe_forge_bidi_indexable' requires an integer size argument}}
    struct StructFoo s;
    (void) forge_bidi_indexable(0, s); // bs-error{{'__unsafe_forge_bidi_indexable' requires an integer size argument}}
    (void) forge_bidi_indexable(0, -4); // bs-error{{negative size argument to '__unsafe_forge_bidi_indexable'}}

    (void) forge_terminated_by(0, f); // expected-error{{'__terminated_by__' attribute requires an integer constant}}
    (void) forge_terminated_by(0, s); // expected-error{{'__terminated_by__' attribute requires an integer constant}}
}

int global_array[6];
void Test4() {
    int local_array[6];
    (void) forge_bidi_indexable(local_array, 0);
    (void) forge_single(local_array);
    (void) forge_terminated_by(local_array, 0);

    (void) forge_bidi_indexable(global_array, 0);
    (void) forge_single(global_array);
    (void) forge_terminated_by(global_array, 0);
}

void Test5() {
    // Function pointers are supported.
    (void) forge_bidi_indexable(Test5, 0);
    (void) forge_single(Test5);
    (void) forge_terminated_by(Test5, 0);
}
