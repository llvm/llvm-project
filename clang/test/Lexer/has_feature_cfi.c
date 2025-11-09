// REQUIRES: target={{x86_64.*-linux.*}}

// RUN: %clang -E -fvisibility=hidden -flto -fno-sanitize-ignorelist -fsanitize=cfi -c %s -o - | FileCheck %s --check-prefix=CHECK-CFI
// RUN: %clang -E -fvisibility=hidden -flto -fno-sanitize-ignorelist -fsanitize=cfi -fsanitize-cfi-cross-dso -c %s -o - | FileCheck %s --check-prefix=CHECK-CFI
// RUN: %clang -E -fvisibility=hidden -flto -fno-sanitize-ignorelist -fsanitize=cfi -fno-sanitize=cfi-nvcall,cfi-vcall,cfi-mfcall,cfi-icall -c %s -o - | FileCheck %s --check-prefix=CHECK-CFI
// CHECK-CFI: CFISanitizerEnabled

// RUN: %clang -E -c %s -o - | FileCheck %s --check-prefix=CHECK-NO-CFI
// CHECK-NO-CFI: CFISanitizerDisabled

// RUN: %clang -E -fsanitize=kcfi -c %s -o - | FileCheck %s --check-prefixes=CHECK-KCFI,CHECK-NO-CFI
// CHECK-KCFI: KCFISanitizerEnabled

// RUN: %clang -E -fsanitize=cfi-cast-strict -c %s -o - | FileCheck %s --check-prefix=CHECK-CFI-CAST-STRICT
// CHECK-CFI-CAST-STRICT: CFICastStrictSanitizerEnabled

// RUN: %clang -E -fvisibility=hidden -flto -fno-sanitize-ignorelist -fsanitize=cfi-derived-cast -c %s -o - | FileCheck %s --check-prefixes=CHECK-CFI,CHECK-CFI-DERIVED-CAST
// CHECK-CFI-DERIVED-CAST: CFIDerivedCastSanitizerEnabled

// RUN: %clang -E -fvisibility=hidden -flto -fno-sanitize-ignorelist -fsanitize=cfi-icall -c %s -o - | FileCheck %s --check-prefixes=CHECK-CFI,CHECK-CFI-ICALL
// CHECK-CFI-ICALL: CFIICallSanitizerEnabled

// RUN: %clang -E -fvisibility=hidden -flto -fno-sanitize-ignorelist -fsanitize=cfi-mfcall -c %s -o - | FileCheck %s --check-prefixes=CHECK-CFI,CHECK-CFI-MFCALL
// CHECK-CFI-MFCALL: CFIMFCallSanitizerEnabled

// RUN: %clang -E -fvisibility=hidden -flto -fno-sanitize-ignorelist -fsanitize=cfi-unrelated-cast -c %s -o - | FileCheck %s --check-prefixes=CHECK-CFI,CHECK-CFI-UNRELATED-CAST
// CHECK-CFI-UNRELATED-CAST: CFIUnrelatedCastSanitizerEnabled

// RUN: %clang -E -fvisibility=hidden -flto -fno-sanitize-ignorelist -fsanitize=cfi-nvcall -c %s -o - | FileCheck %s --check-prefixes=CHECK-CFI,CHECK-CFI-NVCALL
// CHECK-CFI-NVCALL: CFINVCallSanitizerEnabled

// RUN: %clang -E -fvisibility=hidden -flto -fno-sanitize-ignorelist -fsanitize=cfi-vcall -c %s -o - | FileCheck %s --check-prefixes=CHECK-CFI,CHECK-CFI-VCALL
// CHECK-CFI-VCALL: CFIVCallSanitizerEnabled

#if __has_feature(cfi_sanitizer)
int CFISanitizerEnabled();
#else
int CFISanitizerDisabled();
#endif

#if __has_feature(kcfi)
int KCFISanitizerEnabled();
#else
int KCFISanitizerDisabled();
#endif

#if __has_feature(cfi_cast_strict_sanitizer)
int CFICastStrictSanitizerEnabled();
#else
int CFICastStrictSanitizerDisabled();
#endif

#if __has_feature(cfi_derived_cast_sanitizer)
int CFIDerivedCastSanitizerEnabled();
#else
int CFIDerivedCastSanitizerDisabled();
#endif

#if __has_feature(cfi_icall_sanitizer)
int CFIICallSanitizerEnabled();
#else
int CFIICallSanitizerDisabled();
#endif

#if __has_feature(cfi_mfcall_sanitizer)
int CFIMFCallSanitizerEnabled();
#else
int CFIMFCallSanitizerDisabled();
#endif

#if __has_feature(cfi_unrelated_cast_sanitizer)
int CFIUnrelatedCastSanitizerEnabled();
#else
int CFIUnrelatedCastSanitizerDisabled();
#endif

#if __has_feature(cfi_nvcall_sanitizer)
int CFINVCallSanitizerEnabled();
#else
int CFINVCallSanitizerDisabled();
#endif

#if __has_feature(cfi_vcall_sanitizer)
int CFIVCallSanitizerEnabled();
#else
int CFIVCallSanitizerDisabled();
#endif
