// RUN: %clang_cc1 -E %s -triple=aarch64 -fptrauth-intrinsics | \
// RUN:   FileCheck %s --check-prefixes=INTRIN

// RUN: %clang_cc1 -E %s -triple=aarch64 -fptrauth-calls | \
// RUN:   FileCheck %s --check-prefixes=NOINTRIN

// RUN: %clang_cc1 -E %s -DIS_DARWIN -triple=arm64e-apple-darwin -fptrauth-intrinsics | \
// RUN:   FileCheck %s --check-prefixes=INTRIN,INTRIN_MAC

// RUN: %clang_cc1 -E %s -DIS_DARWIN -triple=arm64e-apple-darwin -fptrauth-calls | \
// RUN:   FileCheck %s --check-prefixes=NOINTRIN

#if defined(IS_DARWIN) && __has_extension(ptrauth_qualifier)
// INTRIN_MAC: has_ptrauth_qualifier1
void has_ptrauth_qualifier1() {}
#ifndef __PTRAUTH__
#error ptrauth_qualifier extension present without predefined test macro
#endif
#endif
#if defined(IS_DARWIN) && __has_feature(ptrauth_qualifier)
// INTRIN_MAC: has_ptrauth_qualifier2
void has_ptrauth_qualifier2() {}
#ifndef __PTRAUTH__
#error ptrauth_qualifier extension present without predefined test macro
#endif
#endif
#if defined(__PTRAUTH__)
// INTRIN: has_ptrauth_qualifier3
void has_ptrauth_qualifier3() {}
#endif

#if !defined(__PTRAUTH__) && !__has_feature(ptrauth_qualifier) && !__has_extension(ptrauth_qualifier)
// NOINTRIN: no_ptrauth_qualifier
void no_ptrauth_qualifier() {}
#endif
