// Without '-target' explicitly passed, construct the default triple.
// If the LLVM_DEFAULT_TARGET_TRIPLE is a Darwin triple, change it's architecture
// to a one passed via '-arch'. Otherwise, use '<arch>-apple-darwin10'.

// RUN: %clang -arch x86_64 -c %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix ARCH

// ARCH: "-triple" "x86_64-apple-

// For non-Darwin explicitly passed '-target', ignore '-arch'.

// RUN: %clang -arch arm64 -target x86_64-unknown-linux -c %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix ARCH_NON_DARWIN1

// ARCH_NON_DARWIN1: "-triple" "x86_64-unknown-linux"

// RUN: %clang -arch arm64 -target x86_64-apple -c %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix ARCH_NON_DARWIN2

// ARCH_NON_DARWIN2: "-triple" "x86_64-apple"


// For Darwin explicitly passed '-target', the '-arch' option overrides the architecture

// RUN: %clang -arch arm64 -target x86_64-apple-ios7.0.0 -c %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix ARCH_DARWIN

// ARCH_DARWIN: "-triple" "arm64-apple-ios7.0.0"


// For 'arm64e' and 'arm64e-apple' explicitly passed as '-target',
// construct the default 'arm64e-apple-darwin10' triple.

// RUN: %clang -target arm64e -c %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix ARM64E
// RUN: %clang -target arm64e-apple -c %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix ARM64E
// ARM64E: "-triple" "arm64e-apple-macosx10.6.0"


// For non-Darwin explicitly passed '-target', keep it unchanged if not 'arm64e' and
// 'arm64e-apple', which we implicitly narrow to the default 'arm64e-apple-darwin10'.

// RUN: %clang -target arm64e-pc -c %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix ARM64E_NON_DARWIN1

// ARM64E_NON_DARWIN1: "-triple" "arm64e-pc"

// RUN: %clang -target arm64e-unknown-linux -c %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix ARM64E_NON_DARWIN2

// ARM64E_NON_DARWIN2: "-triple" "arm64e-unknown-linux"


// For Darwin explicitly passed '-target', keep it unchanged

// RUN: %clang -target arm64e-apple-ios7.0.0 -c %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix ARM64E_DARWIN

// ARM64E_DARWIN: "-triple" "arm64e-apple-ios7.0.0"
