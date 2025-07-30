// RUN: %clang_cc1 -E -fsanitize=cfi-cast-strict -o - %s | FileCheck --check-prefix=CHECK-CFISAN %s
// RUN: %clang_cc1 -E -fsanitize=cfi-derived-cast -o - %s | FileCheck --check-prefix=CHECK-CFISAN %s
// RUN: %clang_cc1 -E -fsanitize=cfi-icall -o - %s | FileCheck --check-prefix=CHECK-CFISAN %s
// RUN: %clang_cc1 -E -fsanitize=cfi-mfcall -o - %s | FileCheck --check-prefix=CHECK-CFISAN %s
// RUN: %clang_cc1 -E -fsanitize=cfi-unrelated-cast -o - %s | FileCheck --check-prefix=CHECK-CFISAN %s
// RUN: %clang_cc1 -E -fsanitize=cfi-nvcall -o - %s | FileCheck --check-prefix=CHECK-CFISAN %s
// RUN: %clang_cc1 -E -fsanitize=cfi-vcall -o - %s | FileCheck --check-prefix=CHECK-CFISAN %s
// RUN: %clang_cc1 -E %s -o - | FileCheck --check-prefix=CHECK-NO-CFISAN %s

#if __has_feature(cfi)
int CFISanitizerEnabled();
#else
int CFISanitizerDisabled();
#endif

// CHECK-CFISAN: CFISanitizerEnabled
// CHECK-NO-CFISAN: CFISanitizerDisabled

