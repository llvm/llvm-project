// RUN: %clang_cc1 %s -E -triple=arm64-- -fstack-check | FileCheck %s --check-prefix=ENABLED
// RUN: %clang_cc1 %s -E -triple=arm64-- | FileCheck %s --check-prefix=DISABLED

#if __has_feature(stack_check)
// ENABLED: has_stack_check
void has_stack_check() {}
#else
// DISABLED: not_exists
void not_exists() {}
#endif
