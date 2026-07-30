# REQUIRES: system-linux
#
# Check that BOLT handles TLSDESC -> IE relaxation with stale
# R_AARCH64_TLSDESC_ADD_LO12 on an ADRP (mold linker layout), while
# not regressing the normal GOTTPREL ADRP handling (lld layout).
#
# Build with correct GOTTPREL relocations, then swap the first
# GOTTPREL_PAGE21 to TLSDESC_ADD_LO12 to simulate mold's stale reloc.
#
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-linux %s -o %t.o
# RUN: ld.lld --emit-relocs -shared %t.o -o %t.so
# RUN: obj2yaml %t.so | sed '0,/R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21/{s/R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21/R_AARCH64_TLSDESC_ADD_LO12/}' |\
# RUN: sed '0,/R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21/{s/R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21/R_AARCH64_TLSDESC_ADR_PAGE21/}' | yaml2obj -o %t.mold.so
# RUN: llvm-bolt %t.mold.so -o %t.bolt 2>&1 | FileCheck %s
# RUN: llvm-objdump -d --section=.text %t.bolt | FileCheck %s --check-prefix=DISASM

# CHECK-NOT: BOLT-ERROR

# Both ADRP instructions must target the GOT page, not page 0
# DISASM:      <_start>:
# DISASM-NEXT: adrp x0, 0x{{[1-9][0-9a-f]*}} <tls_var+0x{{[0-9a-f]+}}>
# DISASM:      adrp x0, 0x{{[1-9][0-9a-f]*}} <tls_var+0x{{[0-9a-f]+}}>

  .text
  .globl _start
  .type _start, %function
_start:
  // First TLS access - will become mold-style (TLSDESC_ADD_LO12 on ADRP)
  adrp x0, :gottprel:tls_var
  ldr  x0, [x0, :gottprel_lo12:tls_var]
  mrs  x8, tpidr_el0
  add  x19, x8, x0

  // Second TLS access - stays as lld-style (GOTTPREL on ADRP)
  adrp x0, :gottprel:tls_var
  ldr  x0, [x0, :gottprel_lo12:tls_var]
  mrs  x8, tpidr_el0
  add  x20, x8, x0

  ret
  .size _start, .-_start

  .section .tbss,"awT",@nobits
  .globl tls_var
  .type tls_var, @tls_object
tls_var:
  .word 0
  .size tls_var, 4

