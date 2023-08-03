// RUN: %clang_cc1 %s -E -triple=arm64-- | FileCheck %s --check-prefixes=NOCALLS,NOINTRIN,NORETS,NOQUAL
// RUN: %clang_cc1 %s -E -triple=arm64-- -fptrauth-calls | FileCheck %s --check-prefixes=CALLS,NOINTRIN,NORETS,NOQUAL
// RUN: %clang_cc1 %s -E -triple=arm64-- -fptrauth-returns | FileCheck %s --check-prefixes=NOCALLS,NOINTRIN,RETS,NOQUAL
// RUN: %clang_cc1 %s -E -triple=arm64-- -fptrauth-intrinsics | FileCheck %s --check-prefixes=NOCALLS,INTRIN,NORETS,QUAL

#if __has_feature(ptrauth_calls)
// CALLS: has_ptrauth_calls
void has_ptrauth_calls() {}
#else
// NOCALLS: no_ptrauth_calls
void no_ptrauth_calls() {}
#endif

#if __has_feature(ptrauth_intrinsics)
// INTRIN: has_ptrauth_intrinsics
void has_ptrauth_intrinsics() {}
#else
// NOINTRIN: no_ptrauth_intrinsics
void no_ptrauth_intrinsics() {}
#endif

#if __has_feature(ptrauth_returns)
// RETS: has_ptrauth_returns
void has_ptrauth_returns() {}
#else
// NORETS: no_ptrauth_returns
void no_ptrauth_returns() {}
#endif

#if __has_feature(ptrauth_qualifier)
// QUAL: has_ptrauth_qualifier
void has_ptrauth_qualifier() {}
#else
// NOQUAL: no_ptrauth_qualifier
void no_ptrauth_qualifier() {}
#endif
