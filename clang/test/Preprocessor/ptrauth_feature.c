// RUN: %clang_cc1 %s -E -triple=arm64-- | FileCheck %s --check-prefixes=NOINTRIN
// RUN: %clang_cc1 %s -E -triple=arm64-- -fptrauth-intrinsics | FileCheck %s --check-prefixes=INTRIN

#if __has_feature(ptrauth_intrinsics)
// INTRIN: has_ptrauth_intrinsics
void has_ptrauth_intrinsics() {}
#else
// NOINTRIN: no_ptrauth_intrinsics
void no_ptrauth_intrinsics() {}
#endif
