// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-linux -S -disable-O0-optnone -Werror -Wall -o /dev/null %s
// RUN: %clang_cc1 -triple aarch64-windows -S -disable-O0-optnone -Werror -Wall -o /dev/null %s
// RUN: %clang_cc1 -triple aarch64-darwin -S -disable-O0-optnone -Werror -Wall -o /dev/null %s

#include <stdint.h>

// Ensure that the builtin is defined to take a uint64_t * rather than relying
// on the size of 'unsigned long' which may have different meanings on different
// targets depending on LP64/LLP64.
void test_sme_state_builtin(uint64_t *a,
                            uint64_t *b) __arm_streaming_compatible {
  __builtin_arm_get_sme_state(a, b);
}
