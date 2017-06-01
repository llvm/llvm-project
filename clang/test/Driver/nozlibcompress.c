// RUN: %clang -c %s -Wa,--compress-debug-sections 2>&1 | FileCheck %s
// RUN: %clang -c %s -Wa,--compress-debug-sections -Wa,--nocompress-debug-sections 2>&1 | FileCheck --allow-empty --check-prefix=NOWARN %s
// REQUIRES: nozlib

// CHECK: warning: cannot compress debug sections (zlib not installed)
// NOWARN-NOT: warning: cannot compress debug sections (zlib not installed)
