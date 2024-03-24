# REQUIRES: aarch64
 
# RUN: rm -rf %t && split-file %s %t && cd %t && mkdir be le
# RUN: llvm-mc -filetype=obj -triple=aarch64 main.s -o main.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 ref.s -o le/ref.o && ld.lld -shared le/ref.o -o le/libref.so
# RUN: llvm-mc -filetype=obj -triple=aarch64_be ref.s -o be/ref.o && ld.lld -shared be/ref.o -o be/libref.so
 
# One error formatted library
# RUN: not ld.lld main.o  -lref -Lbe -o /dev/null 2>&1 | FileCheck %s
# RUN: not ld.lld main.o  -l:libref.so -Lbe -o /dev/null 2>&1 | FileCheck %s
 
# Format not specified and the checkFile function is in default.
# RUN: not ld.lld main.o  -lref -Lbe -Lle -o /dev/null 2>&1 | FileCheck %s
# RUN: not ld.lld main.o  -l:libref.so -Lbe -Lle -o /dev/null 2>&1 | FileCheck %s
 
# Format not specified and the checkFile function is closed
# RUN: not ld.lld main.o -no-check-format -lref -Lbe -Lle -o /dev/null 2>&1 | FileCheck %s
# RUN: not ld.lld main.o -no-check-format -l:libref.so -Lbe -Lle -o /dev/null 2>&1 | FileCheck %s
 
# Specified format and the checkFile function is closed
# RUN: not ld.lld -m aarch64linux -no-check-format main.o  -lref -Lbe -Lle -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
# RUN: not ld.lld -m aarch64linux -no-check-format main.o  -l:libref.so -Lbe -Lle -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
 
# Specified format and the checkFile function is in default with one error libary
# RUN: not ld.lld -m aarch64linux main.o  -lref -Lbe -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNTWO
# RUN: not ld.lld -m aarch64linux main.o  -l:libref.so -Lbe -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNONE
 
# Specified format and the checkFile function is in default
# RUN: ld.lld -m aarch64linux  main.o  -lref -Lbe -Lle -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
# RUN: ld.lld -m aarch64linux  main.o  -l:libref.so -Lbe -Lle -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
 
# CHECK-NOT:  error:
# CHECK:      error: be/libref.so is incompatible with main.o
 
# UNONE:      warning: be/libref.so is incompatible with aarch64linux
# UNONE:      error: unable to find library -l:libref.so
# UNTWO:      warning: be/libref.so is incompatible with aarch64linux
# UNTWO:      error: unable to find library -lref
# ERROR:      error: be/libref.so is incompatible with aarch64linux
# WARN:       warning: be/libref.so is incompatible with aarch64linux
 
#--- ref.s
.globl fun
fun:
  nop
#--- main.s
.globl _start
_start:
  bl fun@PLT