// This is similar to the driver test of the same name but we verify that
// the macro is actually set/unset as expected.

// =============================================================================
// Supported target triples
// =============================================================================

// ios
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-ios \
// RUN:  -fno-bounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -Xclang -verify=no_macro -fbounds-safety -fsyntax-only %s
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-ios \
// RUN:  -fbounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -Xclang -verify=macro -fbounds-safety -fsyntax-only %s

// macos
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-macos \
// RUN:  -fno-bounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -Xclang -verify=no_macro -fbounds-safety -fsyntax-only %s
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-macos \
// RUN:  -fbounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -Xclang -verify=macro -fbounds-safety -fsyntax-only %s

// macos (legacy `darwin` os name in triple)
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-darwin \
// RUN:  -fno-bounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -Xclang -verify=no_macro -fbounds-safety -fsyntax-only %s
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-darwin \
// RUN:  -fbounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -Xclang -verify=macro -fbounds-safety -fsyntax-only %s

// watchos
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-watchos \
// RUN:  -fno-bounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -Xclang -verify=no_macro -fbounds-safety -fsyntax-only %s
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-watchos \
// RUN:  -fbounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -Xclang -verify=macro -fbounds-safety -fsyntax-only %s

// tvos
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-tvos \
// RUN:  -fno-bounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -Xclang -verify=no_macro -fbounds-safety -fsyntax-only %s
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-tvos \
// RUN:  -fbounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -Xclang -verify=macro -fbounds-safety -fsyntax-only %s

// =============================================================================
// Unsupported target triples
// =============================================================================

// linux
// RUN: %clang -Wno-incompatible-sysroot -target x86_64-unknown-linux \
// RUN:  -fno-bounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -Xclang -verify=no_macro -fbounds-safety -fsyntax-only %s
// RUN: %clang -Wno-incompatible-sysroot -target x86_64-unknown-linux \
// RUN:  -fbounds-safety-bringup-missing-checks=libc_attributes \
// RUN:  -x c -Xclang -verify=no_macro -fbounds-safety -fsyntax-only %s

// =============================================================================
// Check option is included in all and option variant without `=all` suffix
// =============================================================================

// ios
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-ios \
// RUN:  -fno-bounds-safety-bringup-missing-checks=all \
// RUN:  -x c -Xclang -verify=no_macro -fbounds-safety -fsyntax-only %s
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-ios \
// RUN:  -fno-bounds-safety-bringup-missing-checks \
// RUN:  -x c -Xclang -verify=no_macro -fbounds-safety -fsyntax-only %s

// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-ios \
// RUN:  -fbounds-safety-bringup-missing-checks=all \
// RUN:  -x c -Xclang -verify=macro -fbounds-safety -fsyntax-only %s
// RUN: %clang -Wno-incompatible-sysroot -target arm64-apple-ios \
// RUN:  -fbounds-safety-bringup-missing-checks \
// RUN:  -x c -Xclang -verify=macro -fbounds-safety -fsyntax-only %s

#ifndef __LIBC_STAGED_BOUNDS_SAFETY_ATTRIBUTES
// no_macro-error@+1{{expected __LIBC_STAGED_BOUNDS_SAFETY_ATTRIBUTES macro}}
#error expected __LIBC_STAGED_BOUNDS_SAFETY_ATTRIBUTES macro
#endif

// macro-no-diagnostics
