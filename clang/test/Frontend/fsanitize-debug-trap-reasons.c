// RUN: not %clang_cc1 -triple arm64-apple-macosx14.0.0 \
// RUN: -fsanitize=signed-integer-overflow -fsanitize=signed-integer-overflow \
// RUN: -fsanitize-debug-trap-reasons=bad_value 2>&1 | FileCheck %s

// CHECK: error: invalid value 'bad_value' in '-fsanitize-debug-trap-reasons=bad_value'
int test(void) { return 0;}
