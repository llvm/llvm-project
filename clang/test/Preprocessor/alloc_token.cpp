// RUN: %clang_cc1 -E -fsanitize=alloc-token %s -o - | FileCheck --check-prefix=CHECK-SANITIZE %s
// RUN: %clang_cc1 -E  %s -o - | FileCheck --check-prefix=CHECK-DEFAULT %s

#if __SANITIZE_ALLOC_TOKEN__
// CHECK-SANITIZE: has_sanitize_alloc_token
int has_sanitize_alloc_token();
#else
// CHECK-DEFAULT: no_sanitize_alloc_token
int no_sanitize_alloc_token();
#endif
