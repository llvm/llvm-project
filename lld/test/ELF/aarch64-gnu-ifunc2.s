# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t | FileCheck %s
# RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SEC
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=RELOC

# CHECK:      Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: <myfunc>:
# CHECK-NEXT:   210170:

# CHECK:      <main>:
# .got.plt - page(0x210174) = 0x220190 - 0x210000 = 0x10190
# CHECK-NEXT:   210174: adrp    x8, 0x220000
# CHECK-NEXT:   210178: ldr     x8, [x8, #0x190]
# CHECK-NEXT:   21017c: ret

# CHECK:      Disassembly of section .iplt:
# CHECK-EMPTY:
# CHECK-NEXT: <.iplt>:
# .got.plt - page(0x210180) = 0x220190 - 0x210000 = 0x10190
# CHECK-NEXT:   210180: adrp    x16, 0x220000
# CHECK-NEXT:   210184: ldr     x17, [x16, #0x190]
# CHECK-NEXT:   210188: add     x16, x16, #0x190
# CHECK-NEXT:   21018c: br      x17

# SEC: .got.plt PROGBITS 0000000000220190 000190 000008 00 WA 0 0 8

# RELOC:      Relocations [
# RELOC-NEXT:   Section {{.*}} .rela.dyn {
# RELOC-NEXT:     0x220190 R_AARCH64_IRELATIVE - 0x210170
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
