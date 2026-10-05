# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64 -mattr=+pauth %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=RELA
# RUN: llvm-readelf -x.data %t | FileCheck %s --check-prefix=DATA
# RUN: llvm-readelf -x.got  %t | FileCheck %s --check-prefix=GOT
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s --check-prefix=DIS

## Verify that R_AARCH64_AUTH_ABS64 against a non-preemptible
## undefined weak symbol is resolved to NULL (plus addend).

# RELA-LABEL: Relocations [
# RELA-NEXT:    Section (5) .rela.dyn {
# RELA-NEXT:      0x20400 R_AARCH64_AUTH_GLOB_DAT preempt 0x0
# RELA-NEXT:      0x20408 R_AARCH64_AUTH_TLSDESC preempt 0x0
# RELA-NEXT:      0x30438 R_AARCH64_AUTH_ABS64 preempt 0x0
# RELA-NEXT:      0x30440 R_AARCH64_AUTH_ABS64 preempt 0x25
# RELA-NEXT:      0x30448 R_AARCH64_ABS64 preempt 0x0
# RELA-NEXT:      0x30450 R_AARCH64_ABS64 preempt 0x25
# RELA-NEXT:    }
# RELA-NEXT:  ]

# DATA-LABEL: Hex dump of section '.data':
# DATA-NEXT:  0x00030418 00000000 00000000 25000000 00000000
# DATA-NEXT:  0x00030428 00000000 00000000 25000000 00000000
# DATA-NEXT:  0x00030438 00000000 2a000020 00000000 2a000020
# DATA-NEXT:  0x00030448 00000000 00000000 00000000 00000000

# GOT-LABEL:  Hex dump of section '.got':
# GOT-NEXT:   0x000203f8 00000000 00000000 00000000 000000a0
# GOT-NEXT:   0x00020408 00000000 00000080 00000000 000000a0

# DIS-LABEL:  <_start>:
# DIS-NEXT:     adrp x0, 0x20000
# DIS-NEXT:     ldr  x0, [x0, #0x3f8]
# DIS-NEXT:     mrs  x0, TPIDR_EL0
# DIS-NEXT:     neg  x0, x0
# DIS-NEXT:     nop
# DIS-NEXT:     nop
# DIS-NEXT:     adrp  x0,  0x20000
# DIS-NEXT:     ldr   x0,  [x0, #0x400]
# DIS-NEXT:     adrp  x0,  0x20000
# DIS-NEXT:     ldr   x16, [x0, #0x408]
# DIS-NEXT:     add   x0,  x0, #0x408
# DIS-NEXT:     blraa x16, x0

.weak nonpreempt
.hidden nonpreempt

.weak preempt

.globl _start
_start:
  adrp x0, :got_auth:nonpreempt
  ldr x0, [x0, :got_auth_lo12:nonpreempt]
  adrp  x0,  :tlsdesc_auth:nonpreempt
  ldr   x16, [x0, :tlsdesc_auth_lo12:nonpreempt]
  add   x0,  x0, :tlsdesc_auth_lo12:nonpreempt
  .tlsauthdesccall nonpreempt
  blraa x16, x0
  adrp x0, :got_auth:preempt
  ldr x0, [x0, :got_auth_lo12:preempt]
  adrp  x0,  :tlsdesc_auth:preempt
  ldr   x16, [x0, :tlsdesc_auth_lo12:preempt]
  add   x0,  x0, :tlsdesc_auth_lo12:preempt
  .tlsauthdesccall preempt
  blraa x16, x0

.data
foo:
.quad nonpreempt@AUTH(da,42)
.quad (nonpreempt + 37)@AUTH(da,42)
.quad nonpreempt
.quad (nonpreempt + 37)
.quad preempt@AUTH(da,42)
.quad (preempt + 37)@AUTH(da,42)
.quad preempt
.quad (preempt + 37)
