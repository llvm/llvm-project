// RUN: %clangxx --target=aarch64-unknown-fuchsia -fexperimental-relative-c++-abi-vtables -c %s -### 2>&1 | FileCheck %s --check-prefix=RV
// RUN: %clangxx --target=aarch64-unknown-fuchsia -fno-experimental-relative-c++-abi-vtables -c %s -### 2>&1 | FileCheck %s --check-prefix=NO-RV
// RUN: %clangxx --target=aarch64-unknown-fuchsia -c %s -### 2>&1 | FileCheck %s --check-prefix=NO-RV
// RUN: %clangxx --target=aarch64-unknown-linux-gnu -c %s -### 2>&1 | FileCheck %s --check-prefix=NO-RV

// RV: "-fexperimental-relative-c++-abi-vtables"
// NO-RV-NOT: "-fexperimental-relative-c++-abi-vtables"
