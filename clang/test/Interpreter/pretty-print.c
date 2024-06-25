// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
// RUN: cat %s | clang-repl -Xcc -xc  | FileCheck %s
// RUN: cat %s | clang-repl -Xcc -std=c++11 | FileCheck %s

// Fails with `Symbols not found: [ __clang_Interpreter_SetValueNoAlloc ]`.
// UNSUPPORTED: hwasan

const char* c_str = "Hello, world!"; c_str

// CHECK: Not implement yet.
