

// RUN: %clang_cc1 -fbounds-safety -verify %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#pragma clang abi_ptr_attr set(single)
#pragma clang abi_ptr_attr set(indexable)
#pragma clang abi_ptr_attr set(bidi_indexable)
#pragma clang abi_ptr_attr set(unsafe_indexable)
#pragma clang abi_ptr_attr set(nullable) // expected-error{{'nullable' cannot be set as a default pointer attribute}}
