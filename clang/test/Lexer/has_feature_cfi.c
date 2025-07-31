// RUN: %clang_cc1 -E -fsanitize=cfi-cast-strict -o - %s | FileCheck --check-prefix=CHECK-CFISAN %s
// RUN: %clang_cc1 -E -fsanitize=cfi-derived-cast -o - %s | FileCheck --check-prefix=CHECK-CFISAN %s
// RUN: %clang_cc1 -E -fsanitize=cfi-icall -o - %s | FileCheck --check-prefix=CHECK-CFISAN %s
// RUN: %clang_cc1 -E -fsanitize=cfi-mfcall -o - %s | FileCheck --check-prefix=CHECK-CFISAN %s
// RUN: %clang_cc1 -E -fsanitize=cfi-unrelated-cast -o - %s | FileCheck --check-prefix=CHECK-CFISAN %s
// RUN: %clang_cc1 -E -fsanitize=cfi-nvcall -o - %s | FileCheck --check-prefix=CHECK-CFISAN %s
// RUN: %clang_cc1 -E -fsanitize=cfi-vcall -o - %s | FileCheck --check-prefix=CHECK-CFISAN %s

// RUN: %clang_cc1 -E -fsanitize=kcfi -o - %s | FileCheck --check-prefix=CHECK-CFISAN %s

// RUN: %clang -E --target=x86_64-linux-gnu -fvisibility=hidden -fsanitize=cfi -flto -c %s -o - | FileCheck %s --check-prefix=CHECK-CFISAN
// RUN: %clang -E --target=x86_64-linux-gnu -fvisibility=hidden -fsanitize=cfi -fsanitize-cfi-cross-dso -flto -c %s -o - | FileCheck %s --check-prefix=CHECK-CFISAN

// Disable CFI sanitizers.
// RUN: %clang -E --target=x86_64-linux-gnu -fvisibility=hidden -fno-sanitize=cfi -flto -c %s -o - | FileCheck %s --check-prefix=CHECK-NO-CFISAN

// Disable some but not all CFI schemes.
// RUN: %clang -E --target=x86_64-linux-gnu -fvisibility=hidden -fsanitize=cfi -fno-sanitize-cfi-cross-dso -fno-sanitize=cfi-nvcall,cfi-vcall,cfi-mfcall,cfi-icall -flto -c %s -o - | FileCheck %s --check-prefix=CHECK-CFISAN

// Disable all CFI schemes. This essentially disables CFI sanitizers.
// RUN: %clang -E --target=x86_64-linux-gnu -fvisibility=hidden -fsanitize=cfi -fno-sanitize-cfi-cross-dso -fno-sanitize=cfi-nvcall,cfi-vcall,cfi-mfcall,cfi-icall,cfi-cast-strict,cfi-derived-cast,cfi-unrelated-cast -flto -c %s -o - | FileCheck %s --check-prefix=CHECK-NO-CFISAN

#if __has_feature(cfi)
int CFISanitizerEnabled();
#else
int CFISanitizerDisabled();
#endif

// CHECK-CFISAN: CFISanitizerEnabled
// CHECK-NO-CFISAN: CFISanitizerDisabled

