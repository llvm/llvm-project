// REQUIRES: host-supports-jit, x86_64-linux
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang -std=c++20 %t/mod.cppm --precompile \
// RUN:     -o %t/mod.pcm --target=x86_64-linux-gnu
// RUN: %clang -fPIC %t/mod.pcm -c -o %t/mod.o --target=x86_64-linux-gnu
// RUN: %clang -nostdlib -fPIC -shared %t/mod.o -o %t/libmod.so --target=x86_64-linux-gnu
//
// RUN: cat %t/import.cpp | env LD_LIBRARY_PATH=%t:$LD_LIBRARY_PATH \
// RUN:     clang-repl -Xcc=-std=c++20 -Xcc=-fmodule-file=M=%t/mod.pcm \
// RUN:     -Xcc=--target=x86_64-linux-gnu | FileCheck %t/import.cpp

//--- mod.cppm
export module M;
export const char* Hello() {
    return "Hello Interpreter for Modules!";
}

//--- import.cpp

%lib libmod.so

import M;

extern "C" int printf(const char *, ...);
printf("%s\n", Hello());

// CHECK: Hello Interpreter for Modules!
