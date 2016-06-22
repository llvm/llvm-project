// REQUIRES: x86

// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: not ld.lld -shared %t.o -o %t.so --version-script %p/Inputs/version-script-err.script 2>&1 | FileCheck %s
// CHECK: ; expected, but got }

// RUN: echo    "\"" > %terr1.script
// RUN: not ld.lld --version-script %terr1.script -shared %t.o -o %t.so 2>&1 | \
// RUN:   FileCheck -check-prefix=ERR1 %s
// ERR1: unclosed quote

// RUN: echo "VERSION { extern "C++" {}; }; " > %terr2.script
// RUN: not ld.lld --version-script %terr2.script -shared %t.o -o %t.so 2>&1 | \
// RUN:   FileCheck -check-prefix=ERR2 %s
// ERR2: extern keyword is not supported
