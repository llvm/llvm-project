// RUN: %clang -c %s -### 2>&1 | FileCheck %s --check-prefix=APPLE-CLANG
// RUN: %clang -fintegrated-cc1 -c %s -### 2>&1 | FileCheck %s --check-prefix=EXPLICIT-IN-PROCESS

// REQUIRES: clang-vendor=com.apple.clang

// APPLE-CLANG-NOT: (in-process)
// EXPLICIT-IN-PROCESS: (in-process)
