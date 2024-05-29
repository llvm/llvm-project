// __SVInt8_t is specific to ARM64 so specify that in the target triple
// RUN: %clang_cc1 -triple arm64-apple-darwin -fsyntax-only -verify %s

#define __counted_by(f)  __attribute__((counted_by(f)))

struct on_sizeless_elt_ty {
    int count;
    // expected-error@+2{{'counted_by' only applies to pointers or C99 flexible array members}}
    // expected-error@+1{{array has sizeless element type '__SVInt8_t'}}
    __SVInt8_t arr[] __counted_by(count);
};
