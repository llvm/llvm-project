// UNSUPPORTED: system-windows

// General tests that the header search paths for libstdc++ detected by the
// driver and passed to CC1 are correct on Darwin platforms.

// Check x86 and x86_64
//
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target i686-apple-darwin \
// RUN:     -stdlib=libstdc++ \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_libstdcxx_x86 \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_libstdcxx_x86 --check-prefix=CHECK-LIBSTDCXX-X86 %s
// CHECK-LIBSTDCXX-X86: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-LIBSTDCXX-X86: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1"
// CHECK-LIBSTDCXX-X86: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1/i686-apple-darwin10"
// CHECK-LIBSTDCXX-X86: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1/backward"
// CHECK-LIBSTDCXX-X86: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.0.0"
// CHECK-LIBSTDCXX-X86: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.0.0/i686-apple-darwin8"
// CHECK-LIBSTDCXX-X86: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.0.0/backward"
//
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-apple-darwin \
// RUN:     -stdlib=libstdc++ \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_libstdcxx_x86 \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_libstdcxx_x86 --check-prefix=CHECK-LIBSTDCXX-X86_64 %s
// CHECK-LIBSTDCXX-X86_64: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-LIBSTDCXX-X86_64: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1"
// CHECK-LIBSTDCXX-X86_64: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1/i686-apple-darwin10/x86_64"
// CHECK-LIBSTDCXX-X86_64: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1/backward"
// CHECK-LIBSTDCXX-X86_64: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.0.0"
// CHECK-LIBSTDCXX-X86_64: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.0.0/i686-apple-darwin8"
// CHECK-LIBSTDCXX-X86_64: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.0.0/backward"

// Check arm and thumb
//
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target arm-apple-darwin \
// RUN:     -stdlib=libstdc++ \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_libstdcxx_arm \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_libstdcxx_arm --check-prefix=CHECK-LIBSTDCXX-ARM %s
// CHECK-LIBSTDCXX-ARM: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-LIBSTDCXX-ARM: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1"
// CHECK-LIBSTDCXX-ARM: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1/arm-apple-darwin10/v7"
// CHECK-LIBSTDCXX-ARM: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1/backward"
// CHECK-LIBSTDCXX-ARM: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1"
// CHECK-LIBSTDCXX-ARM: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1/arm-apple-darwin10/v6"
// CHECK-LIBSTDCXX-ARM: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1/backward"
//
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target arm-apple-darwin \
// RUN:     -stdlib=libstdc++ \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_libstdcxx_arm \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_libstdcxx_arm --check-prefix=CHECK-LIBSTDCXX-THUMB %s
// CHECK-LIBSTDCXX-THUMB: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-LIBSTDCXX-THUMB: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1"
// CHECK-LIBSTDCXX-THUMB: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1/arm-apple-darwin10/v7"
// CHECK-LIBSTDCXX-THUMB: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1/backward"
// CHECK-LIBSTDCXX-THUMB: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1"
// CHECK-LIBSTDCXX-THUMB: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1/arm-apple-darwin10/v6"
// CHECK-LIBSTDCXX-THUMB: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1/backward"

// Check aarch64
//
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target arm64-apple-darwin \
// RUN:     -stdlib=libstdc++ \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_libstdcxx_aarch64 \
// RUN:   | FileCheck -DSYSROOT=%S/Inputs/basic_darwin_sdk_libstdcxx_aarch64 --check-prefix=CHECK-LIBSTDCXX-AARCH64 %s
// CHECK-LIBSTDCXX-AARCH64: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-LIBSTDCXX-AARCH64: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1"
// CHECK-LIBSTDCXX-AARCH64: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1/arm64-apple-darwin10"
// CHECK-LIBSTDCXX-AARCH64: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/4.2.1/backward"

// Make sure we issue a warning when we can't find the path
//
// RUN: %clang -no-canonical-prefixes %s -fsyntax-only 2>&1 \
// RUN:     -target x86_64-apple-darwin \
// RUN:     -stdlib=libstdc++ \
// RUN:     -isysroot %S/Inputs/basic_darwin_sdk_no_libstdcxx \
// RUN:   | FileCheck --check-prefix=CHECK-LIBSTDCXX-MISSING %s
// CHECK-LIBSTDCXX-MISSING: clang: warning: include path for libstdc++ headers not found; pass '-stdlib=libc++' on the command line to use the libc++ standard library instead
//
// RUN: %clang %s -target x86_64-apple-darwin -fsyntax-only 2>&1 \
// RUN:           -isysroot %S/Inputs/basic_darwin_sdk_no_libstdcxx \
// RUN:           -stdlib=libc++ \
// RUN:    | FileCheck -allow-empty --check-prefix=CHECK-LIBSTDCXX-MISSING-1 %s
// CHECK-LIBSTDCXX-MISSING-1-NOT: warning
//
// RUN: %clang %s -target x86_64-apple-darwin16 -fsyntax-only 2>&1 \
// RUN:           -isysroot %S/Inputs/basic_darwin_sdk_no_libstdcxx -stdlib=platform \
// RUN:    | FileCheck -allow-empty --check-prefix=CHECK-LIBSTDCXX-MISSING-2 %s
// CHECK-LIBSTDCXX-MISSING-2-NOT: warning
