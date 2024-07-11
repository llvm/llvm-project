// RUN: %clang -### -c -march=x86-64 -msse2avx %s 2>&1 | FileCheck %s

// CHECK: "-msse2avx"
