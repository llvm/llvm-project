
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

void Test(void) {
    void (*fptrImplicitSingleOk)(void) = &Test;
    void (*__single fptrExplicitSingleOk)(void) = &Test;
    void (*__unsafe_indexable fptrExplicitUnsafeIndexableOk)(void) = &Test;
    void (*__indexable fptrExplicitIndexableError)(void) = &Test; // expected-error{{function pointers cannot be indexable}}
    void (*__bidi_indexable fptrExplicitBidiIndexableError)(void) = &Test; // expected-error{{function pointers cannot be indexable}}
    typedef void (*fun_ptr)(void);
    fun_ptr fptrImplicitSingleTypeDefOk = &Test;
    typedef void (*__single fun_ptr_single)(void);

    fun_ptr_single fptrExplicitSingleTypeDefOk1 = &Test;
    fun_ptr __single fptrExplicitSingleTypeDefOk2 = &Test;

    typedef void (*__unsafe_indexable fun_ptr_unsafe)(void);
    fun_ptr_unsafe fptrExplicitUnsafeIndexableTypeDefOk1 = &Test;
    fun_ptr __unsafe_indexable fptrExplicitUnsafeIndexableTypeDefOk2 = &Test;

    typedef void (*__indexable fun_ptr_idxble)(void); // expected-error{{function pointers cannot be indexable}}
    fun_ptr_idxble fptrExplicitIndexableTypeDefError1 = &Test;
    fun_ptr __indexable ptrExplicitIndexableTypeDefError2 = &Test; // expected-error{{function pointers cannot be indexable}}

    typedef void (*__bidi_indexable fun_ptr_bidi_idxble)(void); // expected-error{{function pointers cannot be indexable}}
    fun_ptr_bidi_idxble fptrExplicitBidiIndexableTypeDefError1 = &Test;
    fun_ptr __bidi_indexable fptrExplicitBidiIndexableTypeDefError2 = &Test; // expected-error{{function pointers cannot be indexable}}
}
