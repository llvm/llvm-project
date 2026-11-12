// REQUIRES: webassembly-registered-target

// RUN: %clang -c -target wasm32-unknown-unknown %s -fembed-bitcode -o %t.o
// RUN: llvm-readobj -S %t.o | FileCheck --check-prefix=CHECK %s
// CHECK:   Name: .llvmbc
// CHECK:   Name: .llvmcmd
