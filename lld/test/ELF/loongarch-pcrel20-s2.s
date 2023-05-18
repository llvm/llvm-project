# REQUIRES: loongarch

# RUN: llvm-mc --filetype=obj --triple=loongarch32-unknown-elf %s -o %t.la32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64-unknown-elf %s -o %t.la64.o

# RUN: ld.lld %t.la32.o --section-start=.text=0x20000 --section-start=.data=0x20008 -o %t.la32.1
# RUN: ld.lld %t.la64.o --section-start=.text=0x20000 --section-start=.data=0x20008 -o %t.la64.1
# RUN: llvm-objdump --no-show-raw-insn -d %t.la32.1 | FileCheck --match-full-lines %s
# RUN: llvm-objdump --no-show-raw-insn -d %t.la64.1 | FileCheck --match-full-lines %s
# CHECK: 20000: pcaddi $t0, 2

# RUN: not ld.lld %t.la32.o --section-start=.text=0x20000 --section-start=.data=0x220000 -o /dev/null 2>&1 | \
# RUN:   FileCheck -DFILE=%t.la32.o --check-prefix=ERROR-RANGE %s
# RUN: not ld.lld %t.la64.o --section-start=.text=0x20000 --section-start=.data=0x220000 -o /dev/null 2>&1 | \
# RUN:   FileCheck -DFILE=%t.la64.o --check-prefix=ERROR-RANGE %s
# ERROR-RANGE: error: [[FILE]]:(.text+0x0): relocation R_LARCH_PCREL20_S2 out of range: 2097152 is not in [-2097152, 2097151]; references section '.data'

# RUN: not ld.lld %t.la32.o --section-start=.text=0x20000 --section-start=.data=0x40001 -o /dev/null 2>&1 | \
# RUN:   FileCheck -DFILE=%t.la32.o --check-prefix=ERROR-ALIGN %s
# RUN: not ld.lld %t.la64.o --section-start=.text=0x20000 --section-start=.data=0x40001 -o /dev/null 2>&1 | \
# RUN:   FileCheck -DFILE=%t.la64.o --check-prefix=ERROR-ALIGN %s
# ERROR-ALIGN: error: [[FILE]]:(.text+0x0): improper alignment for relocation R_LARCH_PCREL20_S2: 0x20001 is not aligned to 4 bytes

.global _start

_start:
1:
  pcaddi $t0, 0
  .reloc 1b, R_LARCH_PCREL20_S2, .data

.data
  .word 0
