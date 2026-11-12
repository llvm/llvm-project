// RUN: %clang_cc1 -E -triple x86_64-linux-gnu %s -o - | FileCheck %s

// CHECK: has_noinline_keyword
#if __has_feature(cuda_noinline_keyword)
int has_noinline_keyword();
#else
int no_noinine_keyword();
#endif
