// RUN: %clang_cc1 -fsyntax-only -verify %s -triple x86_64-linux-gnu
// RUN: %clang_cc1 -fsyntax-only -verify %s -triple x86_64-linux-gnu -fexperimental-new-constant-interpreter

// expected-no-diagnostics

// Test that offsetof correctly zero-extends unsigned array indices >= 128.
// Previously, Clang would sign-extend uint8_t indices >= 128, producing
// a large bogus offset value instead of the correct one.
// https://github.com/llvm/llvm-project/issues/199319

#include <stdint.h>
#include <stddef.h>

struct MyStruct {
    void *ptrs[256];
};

_Static_assert(__builtin_offsetof(struct MyStruct, ptrs[(uint8_t)127]) == 127 * sizeof(void *),
               "offsetof with uint8_t index 127 should be correct");

_Static_assert(__builtin_offsetof(struct MyStruct, ptrs[(uint8_t)128]) == 128 * sizeof(void *),
               "offsetof with uint8_t index 128 should be correctly zero-extended, not sign-extended");

_Static_assert(__builtin_offsetof(struct MyStruct, ptrs[(uint8_t)255]) == 255 * sizeof(void *),
               "offsetof with uint8_t index 255 should be correctly zero-extended, not sign-extended");
