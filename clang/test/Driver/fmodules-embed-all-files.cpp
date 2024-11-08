// RUN: %clang -std=c++20 %s -fmodules-embed-all-files -### 2>&1 | FileCheck %s
// CHECK: -fmodules-embed-all-files

// RUN: %clang -std=c++20 %s -### 2>&1 | FileCheck %s --check-prefix=NON-EMBED
// NON-EMBED-NOT: -fmodules-embed-all-files
