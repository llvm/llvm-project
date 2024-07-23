/// Check that clang's idea of runtime dir layout matches e.g. compiler-rt's.

/// Classical runtime layout.
///
/// Cannot be tested: clang -print-runtime-dir always prints the new path even
/// if only the old directories exist.

/// New runtime layout.

// RUN: mkdir -p %t-runtime/lib/x86_64-pc-solaris2.11

/// Canonical triple, 64-bit.
// RUN: %clang -print-runtime-dir --target=x86_64-pc-solaris2.11 \
// RUN:   -resource-dir=%t-runtime \
// RUN:   | FileCheck --check-prefix=RUNTIME-DIR-X86_64 %s

/// Non-canonical triple, 64-bit.
// RUN: %clang -print-runtime-dir --target=amd64-pc-solaris2.11 \
// RUN:   -resource-dir=%t-runtime \
// RUN:   | FileCheck --check-prefix=RUNTIME-DIR-X86_64 %s

// RUNTIME-DIR-X86_64: {{.*}}/lib/x86_64-pc-solaris2.11

// RUN: mkdir -p %t-runtime/lib/i386-pc-solaris2.11

/// Canonical triple, 32-bit.
// RUN: %clang -print-runtime-dir --target=i386-pc-solaris2.11 \
// RUN:   -resource-dir=%t-runtime \
// RUN:   | FileCheck --check-prefix=RUNTIME-DIR-I386 %s

/// Non-canonical triple, 32-bit.
/// clang doesn't normalize --target=i686-pc-solaris2.11 to i386-pc-solaris2.11
/// subdir.

// RUNTIME-DIR-I386: {{.*}}/lib/i386-pc-solaris2.11
