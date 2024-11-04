// Test that -static is forwarded to the linker

// RUN: %clang --target=x86_64-scei-ps4 -static %s -### 2>&1 | FileCheck --check-prefixes=CHECK-STATIC %s

// CHECK-STATIC: {{ld(\.exe)?}}"
// CHECK-STATIC-SAME: "-static"

// Test the driver's control over the JustMyCode behavior with linker flags.

// RUN: %clang --target=x86_64-scei-ps4 -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK-LTO,CHECK-LIB %s
// RUN: %clang --target=x86_64-scei-ps4 -flto=thin -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK-LTO,CHECK-LIB %s
// RUN: %clang --target=x86_64-scei-ps4 -flto=full -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK-LTO,CHECK-LIB %s

// CHECK-LTO: "-lto-debug-options= -enable-jmc-instrument"

// Check the default library name.
// CHECK-LIB: "--whole-archive" "-lSceDbgJmc" "--no-whole-archive"

// Test the driver's control over the -fcrash-diagnostics-dir behavior with linker flags.

// RUN: %clang --target=x86_64-scei-ps4 -fcrash-diagnostics-dir=mydumps %s -### 2>&1 | FileCheck --check-prefixes=CHECK-DIAG-LTO %s
// RUN: %clang --target=x86_64-scei-ps4 -flto=thin -fcrash-diagnostics-dir=mydumps %s -### 2>&1 | FileCheck --check-prefixes=CHECK-DIAG-LTO %s
// RUN: %clang --target=x86_64-scei-ps4 -flto=full -fcrash-diagnostics-dir=mydumps %s -### 2>&1 | FileCheck --check-prefixes=CHECK-DIAG-LTO %s

// CHECK-DIAG-LTO: "-lto-debug-options= -crash-diagnostics-dir=mydumps"

// Test that -lto-debug-options is only supplied to the linker when necessary

// RUN: %clang --target=x86_64-scei-ps4 %s -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-LTO %s
// CHECK-NO-LTO-NOT: -lto-debug-options
