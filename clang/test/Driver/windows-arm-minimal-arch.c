// RUN: not %clang -target thumbv5-windows -mcpu=arm10tdmi %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s

// CHECK: error: the target architecture 'thumbv5' is not supported by the target 'thumbv5-unknown-windows-msvc

