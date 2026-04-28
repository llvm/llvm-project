// Tests that SYCL defaults to C++17 when no -std= is specified

// RUN: %clangxx -### -fsycl -c %s 2>&1 | FileCheck %s --check-prefixes=CHECK-DEVICE,CHECK-HOST

// CHECK-DEVICE: "-cc1"{{.*}} "-fsycl-is-device"
// CHECK-DEVICE-SAME: "-std=c++17"
// CHECK-HOST: "-cc1"{{.*}} "-fsycl-is-host"
// CHECK-HOST-SAME: "-std=c++17"

// Test that explicit -std= overrides the default
// RUN: %clangxx -### -fsycl -std=c++20 -c %s 2>&1 | FileCheck %s --check-prefixes=CHECK-OVERRIDE-DEVICE,CHECK-OVERRIDE-HOST

// CHECK-OVERRIDE-DEVICE: "-cc1"{{.*}} "-fsycl-is-device"
// CHECK-OVERRIDE-DEVICE-SAME: "-std=c++20"
// CHECK-OVERRIDE-DEVICE-NOT: "-std=c++17"
// CHECK-OVERRIDE-HOST: "-cc1"{{.*}} "-fsycl-is-host"
// CHECK-OVERRIDE-HOST-SAME: "-std=c++20"
// CHECK-OVERRIDE-HOST-NOT: "-std=c++17"

// Test that -std=c++14 or earlier produces an error
// RUN: not %clangxx -fsycl -std=c++14 -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-CXX14-ERROR
// RUN: not %clangxx -fsycl -std=c++11 -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-CXX11-ERROR
// RUN: not %clangxx -fsycl -std=c++98 -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-CXX98-ERROR

// CHECK-CXX14-ERROR: error: SYCL requires C++17 or later; '-std=c++14' is not supported
// CHECK-CXX11-ERROR: error: SYCL requires C++17 or later; '-std=c++11' is not supported
// CHECK-CXX98-ERROR: error: SYCL requires C++17 or later; '-std=c++98' is not supported

// Test that C standards produce an error with SYCL
// RUN: not %clangxx -fsycl -std=c11 -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-C11-ERROR
// RUN: not %clangxx -fsycl -std=c99 -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-C99-ERROR

// CHECK-C11-ERROR: error: invalid argument '-std=c11' not allowed with '-fsycl'
// CHECK-C99-ERROR: error: invalid argument '-std=c99' not allowed with '-fsycl'

// Test on Windows with clang-cl (MSVC mode)
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### -fsycl -- %s 2>&1 | FileCheck %s --check-prefixes=CHECK-MSVC-DEVICE,CHECK-MSVC-HOST

// CHECK-MSVC-DEVICE: "-cc1"{{.*}} "-fsycl-is-device"
// CHECK-MSVC-DEVICE-SAME: "-std=c++17"
// CHECK-MSVC-HOST: "-cc1"{{.*}} "-fsycl-is-host"
// CHECK-MSVC-HOST-SAME: "-std=c++17"

// Test that /std: override works on Windows
// RUN: %clang_cl --target=x86_64-pc-windows-msvc -### -fsycl /std:c++20 -- %s 2>&1 | FileCheck %s --check-prefixes=CHECK-MSVC-OVERRIDE-DEVICE,CHECK-MSVC-OVERRIDE-HOST

// CHECK-MSVC-OVERRIDE-DEVICE: "-cc1"{{.*}} "-fsycl-is-device"
// CHECK-MSVC-OVERRIDE-DEVICE-SAME: "-std=c++20"
// CHECK-MSVC-OVERRIDE-DEVICE-NOT: "-std=c++17"
// CHECK-MSVC-OVERRIDE-HOST: "-cc1"{{.*}} "-fsycl-is-host"
// CHECK-MSVC-OVERRIDE-HOST-SAME: "-std=c++20"
// CHECK-MSVC-OVERRIDE-HOST-NOT: "-std=c++17"

// Test that /std:c++14 produces an error on Windows
// RUN: not %clang_cl --target=x86_64-pc-windows-msvc -fsycl /std:c++14 -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-MSVC-CXX14-ERROR

// CHECK-MSVC-CXX14-ERROR: error: SYCL requires C++17 or later; '/std:c++14' is not supported

// Test that C standards produce an error on Windows with clang-cl
// RUN: not %clang_cl --target=x86_64-pc-windows-msvc -fsycl /std:c11 -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-MSVC-C-ERROR

// CHECK-MSVC-C-ERROR: error: invalid argument '/std:c11' not allowed with '-fsycl'

// Test without SYCL - should not default to C++17
// RUN: %clangxx -### -c %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SYCL

// CHECK-NO-SYCL-NOT: "-std=c++17"
