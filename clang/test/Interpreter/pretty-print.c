// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
// RUN: cat %s | clang-repl -Xcc -xc  | FileCheck %s
// RUN: cat %s | clang-repl -Xcc -std=c++11 | FileCheck %s

const char* c_str = "Hello, world!"; c_str

// CHECK: Not implement yet.
