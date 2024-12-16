// Check that when we do CUDA host and device compiles on MacOS, we check for
// includes in /System/Library/Frameworks and /Library/Frameworks.

// RUN: %clang -isysroot /var/empty -target unknown-nvidia-cuda -v -fsyntax-only -x cuda %s -### 2>&1 | FileCheck %s
// CHECK-DAG: "-internal-iframework" "/var/empty/System/Library/Frameworks"
// CHECK-DAG: "-internal-iframework" "/var/empty/Library/Frameworks"
