# REQUIRES: loongarch

# RUN: llvm-mc --filetype=obj --triple=loongarch64-unknown-elf %s -o %t.o

# RUN: ld.lld %t.o --section-start=.text=0x20000 --section-start=.data=0x21000 --section-start=.got=0x21080 -o %t.1
# RUN: llvm-objdump --no-show-raw-insn -d %t.1 | FileCheck --match-full-lines %s
# CHECK: 20000: pcaddu12i $t0, 1
# CHECK: 20004: pcaddu12i $t0, 1

# RUN: not ld.lld %t.o --section-start=.text=0x80021000 --section-start=.data=0x20000 --section-start=.got=0x20080 -o /dev/null 2>&1 | \
# RUN:   FileCheck -DFILE=%t.o --check-prefix=ERROR-RANGE-LOWER %s
# ERROR-RANGE-LOWER: error: [[FILE]]:(.text+0x0): relocation R_LARCH_PCADD_HI20 out of range: -524289 is not in [-524288, 524287]; references section '.data'
# ERROR-RANGE-LOWER: error: [[FILE]]:(.text+0x4): relocation R_LARCH_GOT_PCADD_HI20 out of range: -524289 is not in [-524288, 524287]; references section '.data'

# RUN: not ld.lld %t.o --section-start=.text=0x20000 --section-start=.data=0x8001f800 --section-start=.got=0x8001f880 -o /dev/null 2>&1 | \
# RUN:   FileCheck -DFILE=%t.o --check-prefix=ERROR-RANGE-UPPER %s
# ERROR-RANGE-UPPER: error: [[FILE]]:(.text+0x0): relocation R_LARCH_PCADD_HI20 out of range: 524288 is not in [-524288, 524287]; references section '.data'
# ERROR-RANGE-UPPER: error: [[FILE]]:(.text+0x4): relocation R_LARCH_GOT_PCADD_HI20 out of range: 524288 is not in [-524288, 524287]; references section '.data'

.global _start

_start:
1:
  pcaddu12i $t0, 0
  .reloc 1b, R_LARCH_PCADD_HI20, .data

1:
  pcaddu12i $t0, 0
  .reloc 1b, R_LARCH_GOT_PCADD_HI20, .data

.data
  .word 0
