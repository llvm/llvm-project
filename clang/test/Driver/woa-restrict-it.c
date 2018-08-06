// RUN: %clang -target armv7-windows -### %s 2>&1 | FileCheck %s

// CHECK: "-mllvm" "-arm-restrict-it"

