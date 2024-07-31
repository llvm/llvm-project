// Test the driver's control over the JustMyCode behavior with linker flags.

// RUN: %clang --target=x86_64-scei-ps5 -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-LIB %s
// RUN: %clang --target=x86_64-scei-ps5 -flto -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK-LTO,CHECK-LIB %s

// CHECK-NOT: -plugin-opt=-enable-jmc-instrument
// CHECK-LTO: -plugin-opt=-enable-jmc-instrument

// Check the default library name.
// CHECK-LIB: "--whole-archive" "-lSceJmc_nosubmission" "--no-whole-archive"

// Test the driver's control over the -fcrash-diagnostics-dir behavior with linker flags.

// RUN: %clang --target=x86_64-scei-ps5 -fcrash-diagnostics-dir=mydumps %s -### 2>&1 | FileCheck --check-prefixes=CHECK-DIAG %s
// RUN: %clang --target=x86_64-scei-ps5 -flto -fcrash-diagnostics-dir=mydumps %s -### 2>&1 | FileCheck --check-prefixes=CHECK-DIAG-LTO %s

// CHECK-DIAG-NOT: -plugin-opt=-crash-diagnostics-dir=mydumps
// CHECK-DIAG-LTO: -plugin-opt=-crash-diagnostics-dir=mydumps
