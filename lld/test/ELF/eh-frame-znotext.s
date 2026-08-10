# REQUIRES: aarch64
## While a symbolic relocation for -z notext in .eh_frame can emit a dynamic
## relocation, we try avoiding that (https://github.com/llvm/llvm-project/issues/60392)
## and use a canonical PLT entry instead.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=aarch64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 abi.s -o abi.o
# RUN: ld.lld -shared abi.o -o abi.so

# RUN: ld.lld a.o abi.so -o a
# RUN: llvm-readelf -r --dyn-syms a | FileCheck %s
# RUN: ld.lld -z notext a.o abi.so -o a
# RUN: llvm-readelf -r --dyn-syms a | FileCheck %s

# CHECK: R_AARCH64_JUMP_SLOT {{.*}} __gxx_personality_v0 + 0

# CHECK: 1: 00000000002{{.*}} 0 FUNC    GLOBAL DEFAULT  UND __gxx_personality_v0

#--- a.s
foo:
.cfi_startproc
.cfi_personality 0, __gxx_personality_v0
  ret
.cfi_endproc

#--- abi.s
.globl __gxx_personality_v0
.type __gxx_personality_v0, @function
__gxx_personality_v0:
