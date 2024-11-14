// __SVInt8_t is specific to ARM64 so specify that in the target triple
// RUN: %clang_cc1 -triple arm64-apple-darwin -fsyntax-only -verify %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -triple arm64-apple-darwin -fsyntax-only -verify %s

#define __counted_by_or_null(f)  __attribute__((counted_by_or_null(f)))

struct on_sizeless_pointee_ty {
    int count;
    // expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because '__SVInt8_t' is a sizeless type}}
    __SVInt8_t* member __counted_by_or_null(count);
};

struct on_sizeless_ty {
    int count;
    // expected-error-re@+2{{'counted_by_or_null' only applies to pointers{{$}}}}
    // expected-error@+1{{field has sizeless type '__SVInt8_t'}}
    __SVInt8_t member __counted_by_or_null(count);
};
