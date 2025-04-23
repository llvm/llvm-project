// =============================================================================
// Supported target triples
// =============================================================================

// MACRO: -D__LIBC_STAGED_BOUNDS_SAFETY_ATTRIBUTES
// NO_MACRO-NOT: -D__LIBC_STAGED_BOUNDS_SAFETY_ATTRIBUTES

// ios
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-ios \
// RUN:  -fno-bounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -### -fbounds-safety %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_MACRO %s
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-ios \
// RUN:  -fbounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -### -fbounds-safety %s 2>&1 \
// RUN: | FileCheck --check-prefix=MACRO %s

// macos
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-macos \
// RUN:  -fno-bounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -### -fbounds-safety %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_MACRO %s
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-macos \
// RUN:  -fbounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -### -fbounds-safety %s 2>&1 \
// RUN: | FileCheck --check-prefix=MACRO %s

// macos (legacy `darwin` os name in triple)
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-darwin \
// RUN:  -fno-bounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -### -fbounds-safety %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_MACRO %s
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-darwin \
// RUN:  -fbounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -### -fbounds-safety %s 2>&1 \
// RUN: | FileCheck --check-prefix=MACRO %s

// watchos
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-watchos \
// RUN:  -fno-bounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -### -fbounds-safety %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_MACRO %s
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-watchos \
// RUN:  -fbounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -### -fbounds-safety %s 2>&1 \
// RUN: | FileCheck --check-prefix=MACRO %s

// tvos
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-tvos \
// RUN:  -fno-bounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -### -fbounds-safety %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_MACRO %s
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-tvos \
// RUN:  -fbounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -### -fbounds-safety %s 2>&1 \
// RUN: | FileCheck --check-prefix=MACRO %s

// =============================================================================
// Check option is included in all and option variant without `=all` suffix
// =============================================================================

// ios
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-ios \
// RUN:  -fno-bounds-safety-bringup-missing-checks=all \
// RUN:  -x c -### -fbounds-safety %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_MACRO %s
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-ios \
// RUN:  -fno-bounds-safety-bringup-missing-checks \
// RUN:  -x c -### -fbounds-safety %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_MACRO %s

// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-ios \
// RUN:  -fbounds-safety-bringup-missing-checks=all \
// RUN:  -x c -### -fbounds-safety %s 2>&1 \
// RUN: | FileCheck --check-prefix=MACRO %s
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-ios \
// RUN:  -fbounds-safety-bringup-missing-checks \
// RUN:  -x c -### -fbounds-safety %s 2>&1 \
// RUN: | FileCheck --check-prefix=MACRO %s

// =============================================================================
// Unsupported target triples
// =============================================================================

// linux
// RUN: %clang -Wno-incompatible-sysroot -target x86_64-unknown-linux \
// RUN:  -fno-bounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -### -fbounds-safety %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_MACRO %s
// RUN: %clang -Wno-incompatible-sysroot -target x86_64-unknown-linux \
// RUN:  -fbounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -### -fbounds-safety %s 2>&1 \
// RUN: | FileCheck --check-prefix=NO_MACRO %s
