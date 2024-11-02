// RUN: touch %t.o

// RUN: %clang -target x86_64-apple-driverkit10.15 -isysroot %S/Inputs/MacOSX10.14.sdk -mlinker-version=520 -### %t.o 2>&1 \
// RUN:   | FileCheck %s
// RUN: mkdir -p %t.sdk
// RUN: %clang -target x86_64-apple-driverkit19 -isysroot %t.sdk -mlinker-version=520 -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=MISSING-SDK-JSON-WORKAROUND %s

// RUN: %clang -target arm64-apple-driverkit19 -isysroot %S/Inputs/MacOSX10.14.sdk -fuse-ld= -mlinker-version=520 -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64_NEW %s
// RUN: %clang -target arm64-apple-driverkit19 -isysroot %S/Inputs/MacOSX10.14.sdk -fuse-ld= -mlinker-version=400 -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64_OLD %s
// RUN: %clang -target arm64e-apple-driverkit19 -isysroot %S/Inputs/MacOSX10.14.sdk -fuse-ld= -mlinker-version=520 -### %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=ARM64_NEW %s

// CHECK: "-platform_version" "driverkit" "10.15.0" "10.14"
// MISSING-SDK-JSON-WORKAROUND: "-platform_version" "driverkit" "19.0.0" "19.0.0"

// ARM64_NEW: "-platform_version" "driverkit" "20.0.0" "10.14"
// ARM64_OLD: "-driverkit_version_min" "20.0.0"
