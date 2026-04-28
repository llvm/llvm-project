# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64 -mattr=+pauth %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=RELA
# RUN: llvm-readelf -x.data %t | FileCheck %s --check-prefix=DATA
# RUN: llvm-readelf -x.got  %t | FileCheck %s --check-prefix=GOT
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s --check-prefix=DIS

## Verify that R_AARCH64_AUTH_ABS64 against a weak undefined symbol is resolved
## to NULL (plus addend).

# RELA-LABEL: Relocations [
# RELA-NEXT:  ]

# DATA-LABEL: Hex dump of section '.data':
# DATA-NEXT:  0x00030300 00000000 00000000 25000000 00000000
# DATA-NEXT:  0x00030310 00000000 00000000 25000000 00000000

# GOT-LABEL:  Hex dump of section '.got':
# GOT-NEXT:   0x000202f8 00000000 00000000

# DIS-LABEL:  <_start>:
# DIS-NEXT:     adrp x0, 0x20000
# DIS-NEXT:     ldr  x0, [x0, #0x2f8]
# DIS-NEXT:     mov  x0, #0x0
# DIS-NEXT:     nop
# DIS-NEXT:     nop
# DIS-NEXT:     nop

.weak undef
.hidden undef

.globl _start
_start:
  adrp x0, :got_auth:undef
  ldr x0, [x0, :got_auth_lo12:undef]
  adrp  x0,  :tlsdesc_auth:undef
  ldr   x16, [x0, :tlsdesc_auth_lo12:undef]
  add   x0,  x0, :tlsdesc_auth_lo12:undef
  .tlsdesccall undef
  blraa x16, x0

.data
foo:
.quad undef@AUTH(da,42)
.quad (undef + 37)@AUTH(da,42)
.quad undef
.quad (undef + 37)
