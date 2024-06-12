// RUN: %clang -target arm64-apple-macosx -fptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix PTRAUTH_CALLS
// RUN: %clang -target aarch64-linux-gnu -fptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix PTRAUTH_CALLS
// RUN: %clang -target aarch64-windows-msvc -fptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix PTRAUTH_CALLS
// RUN: not %clang -target x86_64-apple-macosx -fptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix INVALID
// RUN: not %clang -target x86_64-linux-gnu -fptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix INVALID
// RUN: not %clang -target x86_64-windows-msvc -fptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix INVALID

// PTRAUTH_CALLS: "-fptrauth-calls"
// INVALID: unsupported option '-fptrauth-calls'
