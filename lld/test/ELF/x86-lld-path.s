# REQUIRES: x86
 
# RUN: rm -rf %t && split-file %s %t && cd %t && mkdir 32bit 64bit
# RUN: llvm-mc  -filetype=obj -triple=x86_64-unknown-linux-gnu  main.s -o main.o
# RUN: llvm-mc  -filetype=obj -triple=i386-unknown-linux-gnu ref.s -o 32bit/ref.o && ld.lld -shared 32bit/ref.o -o 32bit/libref.so
# RUN: llvm-mc  -filetype=obj -triple=x86_64-unknown-linux-gnu ref.s -o 64bit/ref.o && ld.lld -shared 64bit/ref.o -o 64bit/libref.so
 
# One error formatted library
# RUN: not ld.lld main.o  -lref -L32bit -o /dev/null 2>&1 | FileCheck %s
# RUN: not ld.lld main.o  -l:libref.so -L32bit -o /dev/null 2>&1 | FileCheck %s
 
# Format not specified and the checkFile function is in default.
# RUN: not ld.lld main.o  -lref -L32bit -L64bit -o /dev/null 2>&1 | FileCheck %s
# RUN: not ld.lld main.o  -l:libref.so -L32bit -L64bit -o /dev/null 2>&1 | FileCheck %s
 
# Format not specified and the checkFile function is closed
# RUN: not ld.lld main.o -no-check-format -lref -L32bit -L64bit -o /dev/null 2>&1 | FileCheck %s
# RUN: not ld.lld main.o -no-check-format -l:libref.so -L32bit -L64bit -o /dev/null 2>&1 | FileCheck %s
 
# Specified format and the checkFile function is closed
# RUN: not ld.lld -m elf_x86_64 -no-check-format main.o  -lref -L32bit -L64bit -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
# RUN: not ld.lld -m elf_x86_64 -no-check-format main.o  -l:libref.so -L32bit -L64bit -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
 
# Specified format and the checkFile function is in default with one error libary
# RUN: not ld.lld -m elf_x86_64 main.o  -lref -L32bit -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNTWO
# RUN: not ld.lld -m elf_x86_64 main.o  -l:libref.so -L32bit -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNONE
 
# Specified format and the checkFile function is in default
# RUN: ld.lld -m elf_x86_64  main.o  -lref -L32bit -L64bit -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
# RUN: ld.lld -m elf_x86_64  main.o  -l:libref.so -L32bit -L64bit -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
 
# CHECK-NOT:  error:
# CHECK:      error: 32bit/libref.so is incompatible with main.o
 
# UNONE:      warning: 32bit/libref.so is incompatible with elf_x86_64
# UNONE:      error: unable to find library -l:libref.so
# UNTWO:      warning: 32bit/libref.so is incompatible with elf_x86_64
# UNTWO:      error: unable to find library -lref
# ERROR:      error: 32bit/libref.so is incompatible with elf_x86_64
# WARN:       warning: 32bit/libref.so is incompatible with elf_x86_64
 
#--- ref.s
.globl fun
fun:
  nop
#--- main.s
.global _start
_start:
   call fun
