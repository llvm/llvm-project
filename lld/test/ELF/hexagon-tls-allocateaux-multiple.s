# REQUIRES: hexagon
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf b.s -o b.o
# RUN: ld.lld -shared a.o b.o -o out.so
# RUN: llvm-readobj -r out.so | FileCheck --check-prefix=RELOC %s

#--- a.s
.globl _start
.type _start, @function

_start:
  r2 = add(pc,##_GLOBAL_OFFSET_TABLE_@PCREL)
  r0 = add(r2,##tls_var@GDGOT)
  call tls_var@GDPLT
  jumpr r31

.section .tdata,"awT",@progbits
.globl tls_var
.type tls_var, @object
tls_var:
  .word 0x1234

#--- b.s
.globl other_func
.type other_func, @function

other_func:
  ## Direct call to __tls_get_addr - this creates another path that may
  ## try to allocate auxiliary data for the same symbol
  call __tls_get_addr
  jumpr r31

# RELOC:      Section ({{.*}}) .rela.plt {
# RELOC:        R_HEX_JMP_SLOT __tls_get_addr 0x0
# RELOC:      }
