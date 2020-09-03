# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck -check-prefix=RELOC %s
# RUN: ld.lld %t.o -o %t
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t.so | FileCheck %s
# RUN: llvm-readobj -x .got %t | FileCheck -check-prefix=GOT %s
# RUN: llvm-readobj -x .got %t.so | FileCheck -check-prefix=GOT-SHARED %s
# RUN: llvm-readobj -x .tdata %t | FileCheck -check-prefix=TDATA %s
# RUN: llvm-readobj -x .tdata %t.so | FileCheck -check-prefix=TDATA %s
# RUN: llvm-readobj -r %t | FileCheck -check-prefix=RELA %s
# RUN: llvm-readobj -r %t.so | FileCheck -check-prefix=RELA-SHARED %s

.globl _start
.type _start, @function

_start:
# RELOC:      0x0 R_HEX_GD_GOT_32_6_X a 0x0
# RELOC-NEXT: 0x4 R_HEX_GD_GOT_16_X a 0x0
# CHECK:      {   immext(#0xfffeffc0)
# CHECK-NEXT:     r0 = add(r1,##-0x10008) }
                  r0 = add(r1, ##a@GDGOT)

# RELOC:      0x8 R_HEX_GD_GOT_32_6_X a 0x0
# RELOC-NEXT: 0xC R_HEX_GD_GOT_11_X a 0x0
# CHECK-NEXT: {   immext(#0xfffeffc0)
# CHECK-NEXT:     r0 = memw(r1+##-0x10008) }
                  r0 = memw(r1+##a@GDGOT)

# GOT: Hex dump of section '.got':
# GOT-NEXT: 0x{{[0-9a-f]+}} 01000000 00000000

# GOT-SHARED: Hex dump of section '.got':
# GOT-SHARED-NEXT: 0x{{[0-9a-f]+}} 00000000 00000000

# TDATA: Hex dump of section '.tdata':
# TDATA-NEXT: 01000000

# RELA: Relocations [
# RELA-NEXT: ]

# RELA-SHARED:      .rela.dyn {
# RELA-SHARED-NEXT:   0x2024C R_HEX_DTPMOD_32 a 0x0
# RELA-SHARED-NEXT:   0x20250 R_HEX_DTPREL_32 a 0x0
# RELA-SHARED-NEXT: }

.section        .tdata,"awT",@progbits
.globl  a
a:
.word 1
