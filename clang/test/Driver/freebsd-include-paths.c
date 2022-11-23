// UNSUPPORTED: system-windows

// Check that the driver passes include paths to cc1 on FreeBSD.
// RUN: %clang -### %s --target=x86_64-unknown-freebsd13.1 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DRIVER-PASS-INCLUDES
// DRIVER-PASS-INCLUDES:      "-cc1" {{.*}}"-resource-dir" "[[RESOURCE:[^"]+]]"
// DRIVER-PASS-INCLUDES-SAME: "-internal-isystem" "[[RESOURCE]]/include"
// DRIVER-PASS-INCLUDES-SAME: {{^}} "-internal-externc-isystem" "/usr/include"

// Check that the driver passes include paths to cc1 on FreeBSD in C++ mode.
// RUN: %clang -### -xc++ %s --target=x86_64-unknown-freebsd13.1 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DRIVER-PASS-INCLUDES-CXX
// DRIVER-PASS-INCLUDES-CXX:      "-cc1" {{.*}}"-resource-dir" "[[RESOURCE:[^"]+]]"
// DRIVER-PASS-INCLUDES-CXX-SAME: "-internal-isystem" "/usr/include/c++/v1"
// DRIVER-PASS-INCLUDES-CXX-SAME: {{^}} "-internal-isystem" "[[RESOURCE]]/include"
// DRIVER-PASS-INCLUDES-CXX-SAME: {{^}} "-internal-externc-isystem" "/usr/include"
