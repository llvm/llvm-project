# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readobj -r %t.so | FileCheck %s

## This test verifies that lld handles the case where there's a direct GD_PLT
## relocation against __tls_get_addr. Previously this would create duplicate
## R_HEX_JMP_SLOT relocations for __tls_get_addr. Now only one is created.

# CHECK:      Section ({{.*}}) .rela.dyn {
# CHECK-NEXT:   R_HEX_DTPMOD_32 foo 0x0
# CHECK-NEXT:   R_HEX_DTPREL_32 foo 0x0
# CHECK-NEXT: }
# CHECK:      Section ({{.*}}) .rela.plt {
# CHECK-NEXT:   R_HEX_JMP_SLOT __tls_get_addr 0x0
# CHECK-NEXT: }

.globl _start
.type _start, @function
_start:
  ## Use GD_GOT to set up TLS GOT entry for foo
  r2 = add(pc, ##_GLOBAL_OFFSET_TABLE_@PCREL)
  r0 = add(r2, ##foo@GDGOT)
  ## This creates GD_PLT relocations against __tls_get_addr directly
  call ##__tls_get_addr@GDPLT
  jumpr r31

.section .tdata,"awT",@progbits
.globl foo
.type foo, @object
foo:
  .word 0x11111111
