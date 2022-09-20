// Test the driver's control over the JustMyCode behavior with linker flags.

// RUN: %clang --target=x86_64-scei-ps4 -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK-PS4,CHECK-PS4-LIB %s
// RUN: %clang --target=x86_64-scei-ps4 -flto=thin -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK-PS4-THIN-LTO,CHECK-PS4-LIB %s
// RUN: %clang --target=x86_64-scei-ps4 -flto=full -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK-PS4-FULL-LTO,CHECK-PS4-LIB %s
// RUN: %clang --target=x86_64-scei-ps5 -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK-PS5,CHECK-PS5-LIB %s
// RUN: %clang --target=x86_64-scei-ps5 -flto -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK-PS5-LTO,CHECK-PS5-LIB %s

// CHECK-PS4-NOT: -enable-jmc-instrument

// CHECK-PS4-THIN-LTO: -lto-thin-debug-options=-enable-jmc-instrument
// CHECK-PS4-FULL-LTO: -lto-debug-options=-enable-jmc-instrument

// CHECK-PS5-NOT: "-enable-jmc-instrument"

// CHECK-PS5-LTO: "-mllvm" "-enable-jmc-instrument"

// Check the default library name.
// CHECK-PS4-LIB: "--whole-archive" "-lSceDbgJmc" "--no-whole-archive"
// CHECK-PS5-LIB: "--whole-archive" "-lSceJmc_nosubmission" "--no-whole-archive"
