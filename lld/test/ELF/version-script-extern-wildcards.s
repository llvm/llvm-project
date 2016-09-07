# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: echo "LIBSAMPLE_1.0 { \
# RUN:   global:             \
# RUN:      extern \"C++\" { \
# RUN:        foo*;          \
# RUN:   };                  \
# RUN: };                    \
# RUN: LIBSAMPLE_2.0 {       \
# RUN:   global:             \
# RUN:     extern \"C++\" {  \
# RUN:       zed*;           \
# RUN:       bar;            \
# RUN:   };                  \
# RUN: }; " > %t.script
# RUN: ld.lld --version-script %t.script -shared %t.o -o %t.so
# RUN: llvm-readobj -V -dyn-symbols %t.so | FileCheck %s

# CHECK:  Version symbols {
# CHECK:   Symbols [
# CHECK:    Name: _Z3bari@
# CHECK:    Name: _Z3fooi@@LIBSAMPLE_1.0
# CHECK:    Name: _Z3zedi@@LIBSAMPLE_2.0

.text
.globl _Z3fooi
.type _Z3fooi,@function
_Z3fooi:
retq

.globl _Z3bari
.type _Z3bari,@function
_Z3bari:
retq

.globl _Z3zedi
.type _Z3zedi,@function
_Z3zedi:
retq
