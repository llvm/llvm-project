// RUN: %clang -target unknown-apple-macos10.15 -arch x86_64 -arch x86_64h -arch i386 \
// RUN:   -darwin-target-variant x86_64-apple-ios13.1-macabi -darwin-target-variant x86_64h-apple-ios13.1-macabi \
// RUN:   %s -fuse-ld= -mlinker-version=400 -### 2>&1 | FileCheck %s

// RUN: %clang -target unknown-apple-ios13.1-macabi -arch x86_64 -arch x86_64h \
// RUN:   -darwin-target-variant x86_64-apple-macos10.15 \
// RUN:   %s -fuse-ld= -mlinker-version=400 -### 2>&1 | FileCheck --check-prefix=INVERTED %s

// CHECK: "-arch" "x86_64" "-macosx_version_min" "10.15.0" "-maccatalyst_version_min" "13.1"
// CHECK: "-arch" "x86_64h" "-macosx_version_min" "10.15.0" "-maccatalyst_version_min" "13.1"
// CHECK: "-arch" "i386" "-macosx_version_min" "10.15.0"
// CHECK-NOT: maccatalyst_version_min

// INVERTED: "-arch" "x86_64" "-maccatalyst_version_min" "13.1.0" "-macosx_version_min" "10.15"
// INVERTED: "-arch" "x86_64h" "-maccatalyst_version_min" "13.1.0"
// INVERTED-NOT: macosx_version_min
