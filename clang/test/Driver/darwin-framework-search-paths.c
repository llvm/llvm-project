// UNSUPPORTED: system-windows
//   Windows is unsupported because we use the Unix path separator `/` in the test.

// RUN: %clang %s -target arm64-apple-macosx15.1 -isysroot %S/Inputs/MacOSX15.1.sdk -c %s -### 2>&1 \
// RUN: | FileCheck -DSDKROOT=%S/Inputs/MacOSX15.1.sdk %s
//
// CHECK: "-cc1"
// CHECK: "-resource-dir" "[[RESOURCE_DIR:[^"]*]]"
// CHECK-SAME: "-internal-iframework" "[[SDKROOT]]/System/Library/Frameworks"
// CHECK-SAME: "-internal-iframework" "[[SDKROOT]]/System/Library/SubFrameworks"
// CHECK-SAME: "-internal-iframework" "[[SDKROOT]]/Library/Frameworks"

// Verify that -nostdlibinc and -nostdinc removes the default search paths.
//
// RUN: %clang %s -target arm64-apple-macosx15.1 -isysroot %S/Inputs/MacOSX15.1.sdk -nostdinc -c %s -### 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-NOSTD -DSDKROOT=%S/Inputs/MacOSX15.1.sdk %s
//
// RUN: %clang %s -target arm64-apple-macosx15.1 -isysroot %S/Inputs/MacOSX15.1.sdk -nostdlibinc -c %s -### 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-NOSTD -DSDKROOT=%S/Inputs/MacOSX15.1.sdk %s
//
// CHECK-NOSTD: "-cc1"
// CHECK-NOSTD: "-resource-dir" "[[RESOURCE_DIR:[^"]*]]"
// CHECK-NOSTD-NOT: "-internal-iframework"
