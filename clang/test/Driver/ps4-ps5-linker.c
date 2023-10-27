// Test the driver's control over the JustMyCode behavior with linker flags.

// RUN: %clang --target=x86_64-scei-ps4 -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK-PS4,CHECK-PS4-LIB %s
// RUN: %clang --target=x86_64-scei-ps4 -flto=thin -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK-PS4-THIN-LTO,CHECK-PS4-LIB %s
// RUN: %clang --target=x86_64-scei-ps4 -flto=full -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK-PS4-FULL-LTO,CHECK-PS4-LIB %s
// RUN: %clang --target=x86_64-scei-ps5 -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK-PS5,CHECK-PS5-LIB %s
// RUN: %clang --target=x86_64-scei-ps5 -flto -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK-PS5-LTO,CHECK-PS5-LIB %s

// CHECK-PS4-NOT: -enable-jmc-instrument
// CHECK-PS4-THIN-LTO: "-lto-thin-debug-options= -generate-arange-section -enable-jmc-instrument"
// CHECK-PS4-FULL-LTO: "-lto-debug-options= -generate-arange-section -enable-jmc-instrument"
// CHECK-PS5-NOT: -plugin-opt=-enable-jmc-instrument
// CHECK-PS5-LTO: -plugin-opt=-enable-jmc-instrument

// Check the default library name.
// CHECK-PS4-LIB: "--whole-archive" "-lSceDbgJmc" "--no-whole-archive"
// CHECK-PS5-LIB: "--whole-archive" "-lSceJmc_nosubmission" "--no-whole-archive"

// Test the driver's control over the -fcrash-diagnostics-dir behavior with linker flags.

// RUN: %clang --target=x86_64-scei-ps4 -flto=thin -fcrash-diagnostics-dir=mydumps %s -### 2>&1 | FileCheck --check-prefixes=CHECK-DIAG-PS4-THIN-LTO %s
// RUN: %clang --target=x86_64-scei-ps4 -flto=full -fcrash-diagnostics-dir=mydumps %s -### 2>&1 | FileCheck --check-prefixes=CHECK-DIAG-PS4-FULL-LTO %s
// RUN: %clang --target=x86_64-scei-ps5 -fcrash-diagnostics-dir=mydumps %s -### 2>&1 | FileCheck --check-prefixes=CHECK-DIAG-PS5 %s
// RUN: %clang --target=x86_64-scei-ps5 -flto -fcrash-diagnostics-dir=mydumps %s -### 2>&1 | FileCheck --check-prefixes=CHECK-DIAG-PS5-LTO %s

// CHECK-DIAG-PS4-THIN-LTO: "-lto-thin-debug-options= -generate-arange-section -crash-diagnostics-dir=mydumps"
// CHECK-DIAG-PS4-FULL-LTO: "-lto-debug-options= -generate-arange-section -crash-diagnostics-dir=mydumps"
// CHECK-DIAG-PS5-NOT: -plugin-opt=-crash-diagnostics-dir=mydumps
// CHECK-DIAG-PS5-LTO: -plugin-opt=-crash-diagnostics-dir=mydumps
