// RUN: touch %t.o

// RUN: %clang -target x86_64-apple-ios13.1-macabi -darwin-target-variant x86_64-apple-macos10.15 -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -mlinker-version=520 -### %t.o 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -target x86_64-apple-macos10.14.3 -darwin-target-variant x86_64-apple-ios13.1-macabi -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -mlinker-version=520 -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-INV %s

// RUN: %clang -target arm64-apple-ios13.1-macabi -darwin-target-variant arm64-apple-macos10.15 -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -mlinker-version=520 -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64_NEW %s
// RUN: %clang -target arm64-apple-macos10.15 -darwin-target-variant arm64-apple-ios13.1-macabi  -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -mlinker-version=520 -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64_NEW-INV %s
// RUN: %clang -target arm64-apple-ios13.1-macabi -darwin-target-variant arm64-apple-macos10.15 -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -fuse-ld= -mlinker-version=400 -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64_OLD %s
// RUN: %clang -target arm64-apple-macos10.15 -darwin-target-variant arm64-apple-ios13.1-macabi -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -fuse-ld= -mlinker-version=400 -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64_OLD-INV %s

// CHECK: "-platform_version" "mac catalyst" "13.1.0" "13.1"
// CHECK-SAME: "-platform_version" "macos" "10.15" "10.15"

// CHECK-INV: "-platform_version" "macos" "10.14.3" "10.15"
// CHECK-INV-SAME: "-platform_version" "mac catalyst" "13.1" "13.1"

// ARM64_NEW: "-platform_version" "mac catalyst" "14.0.0" "13.1"
// ARM64_NEW-SAME: "-platform_version" "macos" "11.0.0" "10.15"

// ARM64_NEW-INV: "-platform_version" "macos" "11.0.0" "10.15"
// ARM64_NEW-INV-SAME: "-platform_version" "mac catalyst" "14.0.0" "13.1"

// ARM64_OLD: "-maccatalyst_version_min" "14.0.0" "-macosx_version_min" "11.0.0"
// ARM64_OLD-INV:  "-macosx_version_min" "11.0.0" "-maccatalyst_version_min" "14.0.0"
