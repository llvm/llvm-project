// RUN: not %clang -c -target mips64-linux-gnu -fPIC -mno-abicalls %s 2>&1 | FileCheck %s
// CHECK: error: position-independent code requires ‘-mabicalls’
