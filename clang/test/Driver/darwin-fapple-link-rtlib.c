// RUN: %clang -target arm64-apple-ios12.0 %s -nostdlib -fapple-link-rtlib -resource-dir=%S/Inputs/resource_dir -### 2>&1 | FileCheck %s
// RUN: %clang -target arm64-apple-ios12.0 %s -static -fapple-link-rtlib -resource-dir=%S/Inputs/resource_dir -### 2>&1 | FileCheck %s
// RUN: %clang -target arm64-apple-ios12.0 %s -fapple-link-rtlib -resource-dir=%S/Inputs/resource_dir -### 2>&1 | FileCheck %s --check-prefix=DEFAULT
// CHECK-NOT: "-lSystem"
// DEFAULT: "-lSystem"
// CHECK: libclang_rt.ios.a
