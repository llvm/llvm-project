# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: echo "LIBSAMPLE_2.0 { \
# RUN:   global:             \
# RUN:     extern "C++" {    \
# RUN:       ab[c]*;          \
# RUN:   };                  \
# RUN: }; " > %t.script
# RUN: ld.lld --version-script %t.script -shared %t.o -o %t.so
# RUN: llvm-readobj -V %t.so | FileCheck %s --check-prefix=ABC
# ABC: Name: _Z3abbi@
# ABC: Name: _Z3abci@@LIBSAMPLE_2.0

# RUN: echo "LIBSAMPLE_2.0 { \
# RUN:   global:             \
# RUN:     extern "C++" {    \
# RUN:       ab[b]*;          \
# RUN:   };                  \
# RUN: }; " > %t1.script
# RUN: ld.lld --version-script %t1.script -shared %t.o -o %t1.so
# RUN: llvm-readobj -V %t1.so | FileCheck %s --check-prefix=ABB
# ABB: Name: _Z3abbi@@LIBSAMPLE_2.0
# ABB: Name: _Z3abci@

# RUN: echo "LIBSAMPLE_2.0 { \
# RUN:   global:             \
# RUN:     extern "C++" {    \
# RUN:       ab[a-b]*;       \
# RUN:   };                  \
# RUN: }; " > %t2.script
# RUN: ld.lld --version-script %t2.script -shared %t.o -o %t2.so
# RUN: llvm-readobj -V %t2.so | FileCheck %s --check-prefix=ABB

# RUN: echo "LIBSAMPLE_2.0 { \
# RUN:   global:             \
# RUN:     extern "C++" {    \
# RUN:       ab[a-c]*;       \
# RUN:   };                  \
# RUN: }; " > %t3.script
# RUN: ld.lld --version-script %t3.script -shared %t.o -o %t3.so
# RUN: llvm-readobj -V %t3.so | FileCheck %s --check-prefix=ABBABC
# ABBABC: Name: _Z3abbi@@LIBSAMPLE_2.0
# ABBABC: Name: _Z3abci@@LIBSAMPLE_2.0

# RUN: echo "LIBSAMPLE_2.0 { \
# RUN:   global:             \
# RUN:     extern "C++" {    \
# RUN:       ab[a-bc-d]*;       \
# RUN:   };                  \
# RUN: }; " > %t4.script
# RUN: ld.lld --version-script %t4.script -shared %t.o -o %t4.so
# RUN: llvm-readobj -V %t4.so | FileCheck %s --check-prefix=ABBABC

# RUN: echo "LIBSAMPLE_2.0 { \
# RUN:   global:             \
# RUN:     extern "C++" {    \
# RUN:       ab[a-bd-e]*;       \
# RUN:   };                  \
# RUN: }; " > %t5.script
# RUN: ld.lld --version-script %t5.script -shared %t.o -o %t5.so
# RUN: llvm-readobj -V %t5.so | FileCheck %s --check-prefix=ABB

# RUN: echo "LIBSAMPLE_2.0 { \
# RUN:   global:             \
# RUN:     extern "C++" {    \
# RUN:       ab[^a-c]*;       \
# RUN:   };                  \
# RUN: }; " > %t6.script
# RUN: ld.lld --version-script %t6.script -shared %t.o -o %t6.so
# RUN: llvm-readobj -V %t6.so | FileCheck %s --check-prefix=NO
# NO:  Name: _Z3abbi@
# NO:  Name: _Z3abci@

# RUN: echo "LIBSAMPLE_2.0 { \
# RUN:   global:             \
# RUN:     extern "C++" {    \
# RUN:       ab[^c-z]*;       \
# RUN:   };                  \
# RUN: }; " > %t7.script
# RUN: ld.lld --version-script %t7.script -shared %t.o -o %t7.so
# RUN: llvm-readobj -V %t7.so | FileCheck %s --check-prefix=ABB

# RUN: echo "LIBSAMPLE_2.0 { \
# RUN:   global:             \
# RUN:     extern "C++" {    \
# RUN:       a[x-za-b][a-c]*;       \
# RUN:   };                  \
# RUN: }; " > %t8.script
# RUN: ld.lld --version-script %t8.script -shared %t.o -o %t8.so
# RUN: llvm-readobj -V %t8.so | FileCheck %s --check-prefix=ABBABC

.text
.globl _Z3abci
.type _Z3abci,@function
_Z3abci:
retq

.globl _Z3abbi
.type _Z3abbi,@function
_Z3abbi:
retq
