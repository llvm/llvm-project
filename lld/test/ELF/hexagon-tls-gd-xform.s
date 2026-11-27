# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -defsym GDPLT=1 -triple=hexagon-unknown-elf %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %s -o %t1.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: ld.lld -shared %t1.o -o %t1.so
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t.so | \
# RUN:   FileCheck --check-prefix=CHECK_GDPLT %s
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t1.so | FileCheck %s
# RUN: llvm-readobj -r %t.so | FileCheck -check-prefix=RELA_GDPLT  %s

## Make sure __tls_get_addr is not present unless there is a GDPLT relocation.
# RUN: llvm-readobj -r %t1.so | FileCheck -check-prefix=RELA \
# RUN:   --implicit-check-not="__tls_get_addr" %s

.globl _start
.type _start, @function

_start:
.ifdef GDPLT
                        call x@gdplt
# CHECK_GDPLT:  101e0: { call 0x10210 <__tls_get_addr@plt> }
.else
                  call x
# CHECK:  101b8: { call 0x101e0 <x@plt> }
.endif

# CHECK_GDPLT:        10210: { immext(#0x20040)
# CHECK_GDPLT-NEXT:   10214:   r14 = add(pc,##0x20078) }
# CHECK_GDPLT-NEXT:   10218: { r28 = memw(r14+#0x0) }
# CHECK_GDPLT-NEXT:   1021c: { jumpr r28 }


## Looking at the above check, 0x10210+0x20078 must equal the entry for
##  __tls_get_addr, 0x30288

# RELA_GDPLT: Relocations [
# RELA_GDPLT-NEXT:  Section (5) .rela.plt {
# RELA_GDPLT-NEXT:    0x30288 R_HEX_JMP_SLOT __tls_get_addr 0x0
# RELA_GDPLT-NEXT:  }
# RELA_GDPLT-NEXT:]

# RELA: Relocations [
# RELA-NEXT:  Section (5) .rela.plt {
# RELA-NEXT:    0x30258 R_HEX_JMP_SLOT x 0x0
# RELA-NEXT:  }
# RELA-NEXT:]
