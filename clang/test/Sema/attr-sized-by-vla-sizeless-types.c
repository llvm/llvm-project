// __SVInt8_t is specific to ARM64 so specify that in the target triple
// RUN: %clang_cc1 -triple arm64-apple-darwin -fsyntax-only -verify %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -triple arm64-apple-darwin -fsyntax-only -verify %s

#define __sized_by(f)  __attribute__((sized_by(f)))

struct on_sizeless_elt_ty {
    int count;
    // expected-error-re@+2{{'sized_by' only applies to pointers{{$}}}}
    // expected-error@+1{{array has sizeless element type '__SVInt8_t'}}
    __SVInt8_t arr[] __sized_by(count);
};
