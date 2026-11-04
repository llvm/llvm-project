# REQUIRES: x86
## Test that marker relocations are ignored and undefined symbols lead to errors.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 abi.s -o abi.o
# RUN: ld.lld a.o abi.o -o a
# RUN: llvm-readelf -s a | FileCheck %s

# CHECK: 00000000002{{.*}} 0 FUNC    GLOBAL DEFAULT [[#]] __gxx_personality_v0

# RUN: not ld.lld a.o 2>&1 | FileCheck %s --check-prefix=ERR

# ERR:      error: undefined symbol: __gxx_personality_v0
# ERR-NEXT: >>> referenced by a.o:(.eh_frame+0x12)

#--- a.s
.cfi_startproc
.cfi_personality 0, __gxx_personality_v0
  ret
.cfi_endproc

.section .eh_frame,"a",@unwind
.reloc ., BFD_RELOC_NONE, ignore

#--- abi.s
.globl __gxx_personality_v0
.type __gxx_personality_v0, @function
__gxx_personality_v0:
