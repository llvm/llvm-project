# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readobj -r %t.so --syms | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readobj -r %t.so --syms | FileCheck %s

## Test that we create R_PPC64_RELATIVE for R_PPC64_ADDR64 to non-preemptable
## symbols and R_PPC64_TOC in writable sections.

# CHECK:      .rela.dyn {
# CHECK-NEXT:   0x303C8 R_PPC64_RELATIVE - 0x303C1
## TOC base (0x283b0) + 0x8000 + 1 ---------^
# CHECK-NEXT:   0x303D0 R_PPC64_RELATIVE - 0x303C9
# CHECK-NEXT:   0x303D8 R_PPC64_ADDR64 external 0x1
# CHECK-NEXT:   0x303E0 R_PPC64_ADDR64 global 0x1
# CHECK-NEXT: }
# CHECK-LABEL: Symbols [
# CHECK:       Symbol {
# CHECK:         Name: .TOC. ({{.+}})
# CHECK-NEXT:    Value: 0x283C0

.data
.globl global
global:
local:

.quad .TOC.@tocbase + 1
.quad local + 1
.quad external + 1
.quad global + 1
