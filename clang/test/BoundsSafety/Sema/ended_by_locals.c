
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>
// expected-no-diagnostics

void init_both(int * __bidi_indexable asdf, int asdf_len) {
    const int *myEndPtr = asdf + asdf_len;
    const int * __ended_by(myEndPtr) myEndedByPtr = asdf;
}

void init_end(int * __bidi_indexable asdf, int asdf_len) {
    const int *myEndPtr = asdf + asdf_len;
    const int * __ended_by(myEndPtr) myEndedByPtr; // rdar://103382748 __ended_by needs to be initialized pairwise with end pointer
}

void init_start(int * __bidi_indexable asdf, int asdf_len) {
    const int *myEndPtr; // rdar://103382748 while not breaking bounds safety, this will always trap unless `asdf` is null
    const int * __ended_by(myEndPtr) myEndedByPtr = asdf;
}

void init_neither(int * __bidi_indexable asdf, int asdf_len) {
    const int *myEndPtr;
    const int * __ended_by(myEndPtr) myEndedByPtr;
    myEndPtr = asdf + asdf_len;
    myEndedByPtr = asdf;
}
