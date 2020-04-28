
# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readobj -r %t.so | FileCheck --check-prefix=RELOC %s
# RUN: llvm-readobj -r %t.o | FileCheck --check-prefix=REL %s
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t.so | FileCheck %s

## Prior to D77021 lld would error "relocation R_HEX_GD_PLT_B22_PCREL cannot refer to absolute symbol".
## A PC-relative relocation referencing a non-preemptible absolute symbol (due to STT_TLS) is not representable in -pie/-shared mode.
## For this case we will actually patch the symbol to the external __tls_get_addr which is preemptible.

.globl _start
.type _start, @function

# RELOC:      Section ({{.*}}) .rela.plt {
# RELOC-NEXT:   R_HEX_JMP_SLOT - 0x0
# RELOC-NEXT:   R_HEX_JMP_SLOT __tls_get_addr 0x0
# RELOC-NEXT: }

# REL:      R_HEX_B32_PCREL_X _GLOBAL_OFFSET_TABLE_ 0x0
# REL-NEXT: R_HEX_6_PCREL_X _GLOBAL_OFFSET_TABLE_ 0x4
# REL-NEXT: R_HEX_GD_GOT_32_6_X a 0x0
# REL-NEXT: R_HEX_GD_GOT_16_X a 0x0
# REL-NEXT: R_HEX_GD_PLT_B22_PCREL a 0x0
# REL-NEXT: R_HEX_GD_PLT_B32_PCREL_X a 0x0
# REL-NEXT: R_HEX_GD_PLT_B22_PCREL_X a 0x4

# CHECK:      { immext(#{{.*}})
# CHECK-NEXT:   r2 = add(pc,##{{.*}}) }
# CHECK-NEXT: { immext(#{{.*}})
# CHECK-NEXT:   r0 = add(r2,##-{{.*}}) }
# CHECK-NEXT: { call {{.*}} }
# CHECK-NEXT: {	immext({{.*}})
# CHECK-NEXT:  	call {{.*}} }
# CHECK-NEXT: { r0 = memw(r0+#0x0) }

_start:
  r2 = add(pc,##_GLOBAL_OFFSET_TABLE_@PCREL)
  r0 = add(r2,##a@GDGOT)
  call a@GDPLT
  call ##a@GDPLT
  r0 = memw(r0+#0)

## a is non-preemptible due to STV_HIDDEN visibility.
## We can achieve the same effect with -Bsymbolic.
.section        .tdata,"awT",@progbits
.globl  a
.hidden a
a:
.word 1
