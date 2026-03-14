// RUN: %clang_cc1 -triple x86_64-apple-macosx10.12.0 -fexceptions -std=c++17 -verify %s \
// RUN:   -DEXPECT_DEFINED
//
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.12.0 -fexceptions -std=c++17 -verify %s \
// RUN:   -faligned-alloc-unavailable
//
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.12.0 -fexceptions -std=c++17 -verify %s \
// RUN:   -faligned-allocation -faligned-alloc-unavailable
//
// RUN: %clang_cc1 -triple s390x-none-zos -fexceptions -std=c++17 -verify %s \
// RUN:   -DEXPECT_DEFINED
//
// RUN: %clang_cc1 -triple s390x-none-zos -fexceptions -std=c++17 -verify %s \
// RUN:   -faligned-alloc-unavailable
//
// RUN: %clang_cc1 -triple s390x-none-zos -fexceptions -std=c++17 -verify %s \
// RUN:   -faligned-allocation -faligned-alloc-unavailable

// Test that __cpp_aligned_new is not defined when CC1 is passed
// -faligned-alloc-unavailable by the Darwin and the z/OS driver, even when
// aligned allocation is actually enabled.

// expected-no-diagnostics
#ifdef EXPECT_DEFINED
# ifndef __cpp_aligned_new
#   error "__cpp_aligned_new" should be defined
# endif
#else
# ifdef __cpp_aligned_new
#   error "__cpp_aligned_new" should not be defined
# endif
#endif
