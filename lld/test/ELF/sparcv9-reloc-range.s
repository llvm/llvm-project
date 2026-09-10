# REQUIRES: sparc
# RUN: llvm-mc -filetype=obj -triple=sparcv9 %s -o %t.o
## --threads=1 applies the sections in order, so the diagnostics are ordered.
# RUN: not ld.lld --threads=1 %t.o --defsym=abs=0x10000 --defsym=big=0x100000000000 \
# RUN:   --defsym=far=pc+0x8000000 --defsym=huge=pc+0x200000000 -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --implicit-check-not=error:

## The absolute forms take a signed as well as an unsigned value.
## R_SPARC_HIX22 encodes the complement of the value, which must fit in 32 bits.
## The value it reports is that complement, here ~0x10000.
# CHECK: error: {{.*}}:(.abs+0x0): relocation R_SPARC_8 out of range: 65536 is not in [-128, 255]; references 'abs'
# CHECK: error: {{.*}}:(.abs+0x2): relocation R_SPARC_16 out of range: 65536 is not in [-32768, 65535]; references 'abs'
# CHECK: error: {{.*}}:(.abs+0x4): relocation R_SPARC_13 out of range: 65536 is not in [-4096, 8191]; references 'abs'
# CHECK: error: {{.*}}:(.abs+0x8): relocation R_SPARC_HI22 out of range: 17592186044416 is not in [0, 4294967295]; references 'big'
# CHECK: error: {{.*}}:(.abs+0xc): relocation R_SPARC_H44 out of range: 17592186044416 is not in [0, 17592186044415]; references 'big'
# CHECK: error: {{.*}}:(.abs+0x10): relocation R_SPARC_HIX22 out of range: 18446744073709486079 is not in [0, 4294967295]; references 'abs'
.section .abs,"ax",@progbits
  .byte abs
  .balign 2
  .half abs
  .balign 4
  or    %g0, abs, %g1
  sethi %hi(big), %g1
  sethi %h44(big), %g1
  sethi %hix(abs), %g1

## The PC-relative forms are signed, except R_SPARC_PC22, which is a bit field.
## far and huge are relative to pc, so the displacements do not depend on the
## output layout.
# CHECK: error: {{.*}}:(.pcrel+0x0): relocation R_SPARC_DISP8 out of range: 134217728 is not in [-128, 127]; references 'far'
# CHECK: error: {{.*}}:(.pcrel+0x1): relocation R_SPARC_DISP16 out of range: 134217727 is not in [-32768, 32767]; references 'far'
# CHECK: error: {{.*}}:(.pcrel+0x3): relocation R_SPARC_DISP32 out of range: 8589934589 is not in [-2147483648, 2147483647]; references 'huge'
# CHECK: error: {{.*}}:(.pcrel+0x8): relocation R_SPARC_WDISP16 out of range: 134217720 is not in [-131072, 131071]; references 'far'
# CHECK: error: {{.*}}:(.pcrel+0xc): relocation R_SPARC_WDISP19 out of range: 134217716 is not in [-1048576, 1048575]; references 'far'
# CHECK: error: {{.*}}:(.pcrel+0x10): relocation R_SPARC_WDISP22 out of range: 134217712 is not in [-8388608, 8388607]; references 'far'
# CHECK: error: {{.*}}:(.pcrel+0x14): relocation R_SPARC_WDISP30 out of range: 8589934572 is not in [-2147483648, 2147483647]; references 'huge'
# CHECK: error: {{.*}}:(.pcrel+0x18): relocation R_SPARC_PC22 out of range: 8589934568 is not in [-2147483648, 4294967295]; references 'huge'
.section .pcrel,"ax",@progbits
.globl pc
pc:
  .byte far - .
  .half far - .
  .word huge - .
  .balign 4
  brnz  %g1, far
  be,pt %icc, far
  ba    far
  call  huge
  sethi %pc22(huge), %g5
