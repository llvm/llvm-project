# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: echo "FOO { global: extern \"C++\" { \"aaa*\"; }; };" > %t.script
# RUN: ld.lld --version-script %t.script -shared %t.o -o %t.so
# RUN: llvm-readobj -V -dyn-symbols %t.so | FileCheck %s

# CHECK: Symbol {
# CHECK:   Name: _Z3aaaPf@ (1)
# CHECK: Symbol {
# CHECK:   Name: _Z3aaaPi@ (10)

.text
.globl _Z3aaaPi
.type _Z3aaaPi,@function
_Z3aaaPi:
retq

.globl _Z3aaaPf
.type _Z3aaaPf,@function
_Z3aaaPf:
retq
