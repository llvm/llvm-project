// REQUIRES: host-supports-jit, x86_64-linux
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: cat %t/inline-asm.txt | clang-repl -Xcc="-I%t"

//--- inline-asm.cpp
__asm(".globl _ZSt21ios_base_library_initv");
int x;

//--- inline-asm.txt
#include "inline-asm.cpp"
x = 10;
%quit
