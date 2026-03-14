// RUN: %clang -### -target aarch64-unknown-windows-msvc -c %s 2>&1 | FileCheck -check-prefix CHECK-UNSPECIFIED %s
// RUN: %clang -### -target thumbv7-unknown-linux-android -c %s 2>&1 | FileCheck -check-prefix CHECK-UNSPECIFIED %s
// RUN: %clang -### -target x86_64-apple-macosx -c %s 2>&1 | FileCheck -check-prefix CHECK-UNSPECIFIED %s

// RUN: %clang -### -target aarch64-unknown-windows-msvc -fcf-runtime-abi=objc -c %s 2>&1 | FileCheck -check-prefix CHECK-OBJC %s
// RUN: %clang -### -target thumbv7-unknown-linux-android -fcf-runtime-abi=swift-5.0 -c %s 2>&1 | FileCheck -check-prefix CHECK-SWIFT-5_0 %s
// RUN: %clang -### -target i386-unknown-freebsd -fcf-runtime-abi=swift-4.2 -c %s 2>&1 | FileCheck -check-prefix CHECK-SWIFT-4_2 %s
// RUN: %clang -### -target s390x-unknown-linux-gnu -fcf-runtime-abi=swift-4.1 -c %s 2>&1 | FileCheck -check-prefix CHECK-SWIFT-4_1 %s
// RUN: %clang -### -target x86_64-apple-macosx -fcf-runtime-abi=swift -c %s 2>&1 | FileCheck -check-prefix CHECK-SWIFT %s

// RUN: not %clang -### -target arm7k-apple-watchos -fcf-runtime-abi=invalid -c %s 2>&1 | FileCheck -check-prefix CHECK-INVALID %s

// CHECK-UNSPECIFIED-NOT: "-fcf-runtime-abi=

// CHECK-OBJC: "-fcf-runtime-abi=objc"
// CHECK-SWIFT-5_0: "-fcf-runtime-abi=swift-5.0"
// CHECK-SWIFT-4_2: "-fcf-runtime-abi=swift-4.2"
// CHECK-SWIFT-4_1: "-fcf-runtime-abi=swift-4.1"
// CHECK-SWIFT: "-fcf-runtime-abi=swift"

// CHECK-INVALID: error: invalid CoreFoundation Runtime ABI 'invalid'; must be one of 'objc', 'standalone', 'swift', 'swift-5.0', 'swift-4.2', 'swift-4.1'

