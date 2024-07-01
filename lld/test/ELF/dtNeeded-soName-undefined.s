# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=aarch64 main.s -o main.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 def.s -o def.o && ld.lld -shared def.o -o def.so
# RUN: llvm-mc -filetype=obj -triple=aarch64 ref.s -o ref.o && ld.lld -shared ref.o -o ref.so ./def.so

# RUN: not ld.lld main.o ref.so def.so -o /dev/null 2>&1 | FileCheck %s

# CHECK-NOT:  error:
# CHECK:      error: undefined reference due to --no-allow-shlib-undefined: foo
# CHECK-NEXT: >>> referenced by ref.so


#--- def.s

#--- ref.s
.globl fun
fun:
  bl foo@PLT
#--- main.s
.globl _start
_start:
  bl fun@PLT
