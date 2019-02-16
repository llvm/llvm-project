// REQUIRES: arm-registered-target

// RUN: %clang -c -target armv7-apple-ios10 %s -fembed-bitcode -### 2>&1 | FileCheck %s -check-prefix=CHECK-AS
// RUN: %clang -c -target armv7-apple-ios10 %s -fembed-bitcode-marker -### 2>&1 | FileCheck %s -check-prefix=CHECK-AS-MARKER
// CHECK-AS: -cc1as
// CHECK-AS: -fembed-bitcode=all
// CHECK-AS-MARKER: -fembed-bitcode=marker

// RUN: %clang -c -target armv7-apple-ios10 %s -fembed-bitcode -o %t.o
// RUN: llvm-readobj -section-headers %t.o | FileCheck --check-prefix=CHECK-SECTION %s
// CHECK-SECTION: Name: __asm
// CHECK-SECTION-NEXT: Segment: __LLVM
