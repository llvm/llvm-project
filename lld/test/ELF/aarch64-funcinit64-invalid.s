# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
# RUN: not ld.lld %t.o -o %t 2>&1 | FileCheck --check-prefix=ERR %s

.rodata
# ERR: relocation R_AARCH64_FUNCINIT64 cannot be used against local symbol
.8byte func@FUNCINIT

.data
# ERR: relocation R_AARCH64_FUNCINIT64 cannot be used against ifunc symbol 'ifunc'
.8byte ifunc@FUNCINIT

.text
func:
.type ifunc, @gnu_indirect_function
ifunc:
ret
