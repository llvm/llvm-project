// RUN: ml /c /coff /Fo%t_asm.obj %p/intercept_detour_minus_one.asm
// RUN: %clang_cl -Od %s %t_asm.obj -Fe%t /link /INFERASANLIBS
// RUN: %run %t 2>&1 | FileCheck %s

// This test is for the Windows 32bit-specific interception technique detour. 
// There is a rare instance of a short jump containing a 0xCC offset placed
// right before 4 bytes of 0xCC padding. This test checks that the
// detour function override is not applied in that instance.
// UNSUPPORTED: asan-64-bits

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <algorithm>

extern "C" __declspec(dllimport)
bool __cdecl __sanitizer_override_function_by_addr(
    void *source_function,
    void *target_function,
    void **old_target_function = nullptr
    );

template <typename F>
F *apply_interception(const F& source, const F& target) {
    void *old_default = nullptr;
    if (!__sanitizer_override_function_by_addr(&source, &target, &old_default)) {
        fputs("__sanitizer_override_function_by_addr failed.", stderr); // CHECK-NOT: __sanitizer_override_function_by_addr failed.
        exit(1);
    }
    return reinterpret_cast<F*>(old_default);
}

extern "C" bool validate_interception(void * addr) {
    // Checks if 5 preceding bytes have been
    // corrupted by the interception.
    auto p = static_cast<const uint8_t*>(addr); // use uint8_t for byte-wise access
    return std::all_of(p - 5, p, [](uint8_t byte) {
        // 0xCC: INT3 (breakpoint instruction), used for padding or debugging
        // 0x90: NOP (no operation), used for padding or alignment
        return byte == 0xCC || byte == 0x90;
    });
}

int dummy_function() {
    // Dummy function to overriding with.
    // It should not be called directly in this test.
    return 0;
}
extern "C" int false_header();
extern "C" int function_to_intercept();

int main() {
    auto func = apply_interception(dummy_function, function_to_intercept);
    if (validate_interception((void*)function_to_intercept)) {
        printf("Success\n"); // CHECK: Success
        return 0;
    }
    return 1;
}