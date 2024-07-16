// Test the driver's control over the JustMyCode behavior with linker flags.

// RUN: %clang --target=x86_64-scei-ps4 -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-LIB %s
// RUN: %clang --target=x86_64-scei-ps4 -flto=thin -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK-THIN-LTO,CHECK-LIB %s
// RUN: %clang --target=x86_64-scei-ps4 -flto=full -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK-FULL-LTO,CHECK-LIB %s

// CHECK-NOT: -enable-jmc-instrument
// CHECK-THIN-LTO: "-lto-thin-debug-options= -generate-arange-section -enable-jmc-instrument"
// CHECK-FULL-LTO: "-lto-debug-options= -generate-arange-section -enable-jmc-instrument"

// Check the default library name.
// CHECK-LIB: "--whole-archive" "-lSceDbgJmc" "--no-whole-archive"

// Test the driver's control over the -fcrash-diagnostics-dir behavior with linker flags.

// RUN: %clang --target=x86_64-scei-ps4 -flto=thin -fcrash-diagnostics-dir=mydumps %s -### 2>&1 | FileCheck --check-prefixes=CHECK-DIAG-THIN-LTO %s
// RUN: %clang --target=x86_64-scei-ps4 -flto=full -fcrash-diagnostics-dir=mydumps %s -### 2>&1 | FileCheck --check-prefixes=CHECK-DIAG-FULL-LTO %s

// CHECK-DIAG-THIN-LTO: "-lto-thin-debug-options= -generate-arange-section -crash-diagnostics-dir=mydumps"
// CHECK-DIAG-FULL-LTO: "-lto-debug-options= -generate-arange-section -crash-diagnostics-dir=mydumps"
