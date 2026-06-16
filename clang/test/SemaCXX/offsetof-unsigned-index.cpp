// RUN: %clang_cc1 -fsyntax-only -verify %s -triple x86_64-linux-gnu -std=c++11
// RUN: %clang_cc1 -fsyntax-only -verify %s -triple x86_64-linux-gnu -std=c++11 -fexperimental-new-constant-interpreter

// expected-no-diagnostics

// Test that offsetof correctly zero-extends unsigned array indices >= 128.
// Previously, Clang would sign-extend uint8_t/uint16_t indices whose high bit
// was set, producing a large bogus offset value instead of the correct one.
// https://github.com/llvm/llvm-project/issues/199319

#include <cstdint>
#include <cstddef>

struct MyStruct {
    void *ptrs[256];
};

// uint8_t index: values >= 128 were incorrectly sign-extended
static_assert(__builtin_offsetof(MyStruct, ptrs[(uint8_t)127]) == 127 * sizeof(void *),
              "offsetof with uint8_t index 127 should be correct");
static_assert(__builtin_offsetof(MyStruct, ptrs[(uint8_t)128]) == 128 * sizeof(void *),
              "offsetof with uint8_t index 128 should be correctly zero-extended");
static_assert(__builtin_offsetof(MyStruct, ptrs[(uint8_t)255]) == 255 * sizeof(void *),
              "offsetof with uint8_t index 255 should be correctly zero-extended");

// uint16_t index: values >= 32768 were also affected
struct BigStruct {
    char data[65536];
};
static_assert(__builtin_offsetof(BigStruct, data[(uint16_t)32768]) == 32768,
              "offsetof with uint16_t index 32768 should be correctly zero-extended");
