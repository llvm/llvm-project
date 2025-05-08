// RUN: %clang_cc1 -E %s -triple=aarch64 -fptrauth-intrinsics | \
// RUN:   FileCheck %s --check-prefixes=INTRIN

// RUN: %clang_cc1 -E %s -triple=aarch64 -fptrauth-calls | \
// RUN:   FileCheck %s --check-prefixes=NOINTRIN

#if __has_extension(ptrauth_qualifier)
// INTRIN: has_ptrauth_qualifier
void has_ptrauth_qualifier() {}
#else
// NOINTRIN: no_ptrauth_qualifier
void no_ptrauth_qualifier() {}
#endif
