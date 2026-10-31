// REQUIRES: host-supports-jit, x86_64-linux
//
// clang-repl compiles position-independent code (it injects -fPIC), so a PCM
// (Clang module) built with a different PIC level is incompatible and must be
// rejected instead of silently accepted as a "compatible" language-option
// difference.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang -fno-pic -std=c++20 %t/mod.cppm --precompile \
// RUN:     -o %t/mod.pcm --target=x86_64-linux-gnu
//
// RUN: echo '// empty' \
// RUN:     | not clang-repl -Xcc=-std=c++20 -Xcc=-fmodule-file=M=%t/mod.pcm \
// RUN:           -Xcc=--target=x86_64-linux-gnu 2>&1 \
// RUN:     | FileCheck %s

//--- mod.cppm
export module M;
export const char *Hello() { return "Hello Interpreter for Modules!"; }

// CHECK: incompatible with clang-repl's PIC level
