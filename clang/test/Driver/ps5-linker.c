// Test that PIE is the default for main components

// RUN: %clang --target=x86_64-sie-ps5 %s -### 2>&1 | FileCheck --check-prefixes=CHECK-PIE %s

// CHECK-PIE: {{ld(\.exe)?}}"
// CHECK-PIE-SAME: "-pie"

// RUN: %clang --target=x86_64-sie-ps5 -no-pie %s -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-PIE %s
// RUN: %clang --target=x86_64-sie-ps5 -r %s -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-PIE %s
// RUN: %clang --target=x86_64-sie-ps5 -shared %s -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-PIE,CHECK-SHARED %s
// RUN: %clang --target=x86_64-sie-ps5 -static %s -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-PIE %s

// CHECK-NO-PIE: {{ld(\.exe)?}}"
// CHECK-NO-PIE-NOT: "-pie"
// CHECK-SHARED: "--shared"

// Test that -static is forwarded to the linker

// RUN: %clang --target=x86_64-sie-ps5 -static %s -### 2>&1 | FileCheck --check-prefixes=CHECK-STATIC %s

// CHECK-STATIC: {{ld(\.exe)?}}"
// CHECK-STATIC-SAME: "-static"

// Test the driver's control over the JustMyCode behavior with linker flags.

// RUN: %clang --target=x86_64-sie-ps5 -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-LIB %s
// RUN: %clang --target=x86_64-sie-ps5 -flto -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-LIB %s

// CHECK: -plugin-opt=-enable-jmc-instrument

// Check the default library name.
// CHECK-LIB: "--whole-archive" "-lSceJmc_nosubmission" "--no-whole-archive"

// Test the driver's control over the -fcrash-diagnostics-dir behavior with linker flags.

// RUN: %clang --target=x86_64-sie-ps5 -fcrash-diagnostics-dir=mydumps %s -### 2>&1 | FileCheck --check-prefixes=CHECK-DIAG %s
// RUN: %clang --target=x86_64-sie-ps5 -flto -fcrash-diagnostics-dir=mydumps %s -### 2>&1 | FileCheck --check-prefixes=CHECK-DIAG %s

// CHECK-DIAG: -plugin-opt=-crash-diagnostics-dir=mydumps
