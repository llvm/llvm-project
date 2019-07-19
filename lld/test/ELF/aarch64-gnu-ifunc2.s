# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux-gnu %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
# RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SEC
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=RELOC

# CHECK:      Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: myfunc:
# CHECK-NEXT:   210000:

# CHECK:      main:
# adrp x8, 0x220000, 0x220000 == address in .got.plt
# CHECK-NEXT:   210004: adrp    x8, #65536
# CHECK-NEXT:   210008: ldr     x8, [x8]
# CHECK-NEXT:   21000c: ret

# CHECK:      Disassembly of section .plt:
# CHECK-EMPTY:
# CHECK-NEXT: .plt:
# adrp x16, 0x220000, 0x220000 == address in .got.plt
# CHECK-NEXT:   210010: adrp    x16, #65536
# CHECK-NEXT:   210014: ldr     x17, [x16]
# CHECK-NEXT:   210018: add     x16, x16, #0
# CHECK-NEXT:   21001c: br      x17

# SEC: .got.plt PROGBITS 0000000000220000 020000 000008 00 WA 0 0 8

# RELOC:      Relocations [
# RELOC-NEXT:   Section {{.*}} .rela.plt {
# RELOC-NEXT:     0x220000 R_AARCH64_IRELATIVE - 0x210000
# RELOC-NEXT:   }
# RELOC-NEXT: ]

.text
.globl myfunc
.type myfunc,@gnu_indirect_function
myfunc:
 ret

.text
.globl main
.type main,@function
main:
 adrp x8, :got:myfunc
 ldr  x8, [x8, :got_lo12:myfunc]
 ret
