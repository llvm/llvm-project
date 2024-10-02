// __SVInt8_t is specific to ARM64 so specify that in the target triple
// RUN: %clang_cc1 -triple arm64-apple-darwin -fsyntax-only -verify %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -triple arm64-apple-darwin -fsyntax-only -verify %s

#define __sized_by_or_null(f)  __attribute__((sized_by_or_null(f)))

struct on_sizeless_pointee_ty {
    int size;
    // expected-error@+1{{'sized_by_or_null' cannot be applied to a pointer with pointee of unknown size because '__SVInt8_t' is a sizeless type}}
    __SVInt8_t* member __sized_by_or_null(size);
};

struct on_sizeless_ty {
    int size;
    // expected-error-re@+2{{'sized_by_or_null' only applies to pointers{{$}}}}
    // expected-error@+1{{field has sizeless type '__SVInt8_t'}}
    __SVInt8_t member __sized_by_or_null(size);
};
