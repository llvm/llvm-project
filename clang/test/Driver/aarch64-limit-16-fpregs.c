// RUN: %clang -target aarch64-none-gnu -mlimit-16-fpregs -### %s 2> %t
// RUN: FileCheck < %t %s

// CHECK: "-target-feature" "+limit-16-fpregs"
