# REQUIRES: loongarch

# RUN: llvm-mc --filetype=obj --triple=loongarch64-unknown-elf %s -o %t.o

# RUN: ld.lld %t.o --section-start=.text=0x20010 --section-start=.sec.foo=0x60020 -o %t1
# RUN: llvm-objdump --no-show-raw-insn -d %t1 | FileCheck --match-full-lines %s --check-prefix=CASE1
## hi20 = target - pc + (1 << 17) >> 18 = 0x60020 - 0x20010 + 0x20000 >> 18 = 1
## lo18 = target - pc & (1 << 18) - 1 = 0x60020 - 0x20010 & 0x3ffff = 16
# CASE1:      20010: pcaddu18i $ra, 1
# CASE1-NEXT: 20014: jirl $zero, $ra, 16

# RUN: ld.lld %t.o --section-start=.text=0x20010 --section-start=.sec.foo=0x40020 -o %t2
# RUN: llvm-objdump --no-show-raw-insn -d %t2 | FileCheck --match-full-lines %s --check-prefix=CASE2
## hi20 = target - pc + (1 << 17) >> 18 = 0x40020 - 0x20010 + 0x20000 >> 18 = 1
## lo18 = target - pc & (1 << 18) - 1 = 0x40020 - 0x20010 & 0x3ffff = -131056
# CASE2:      20010: pcaddu18i $ra, 1
# CASE2-NEXT: 20014: jirl $zero, $ra, -131056

# RUN: not ld.lld %t.o --section-start=.text=0x20000 --section-start=.sec.foo=0x2000020000 -o /dev/null 2>&1 | \
# RUN:   FileCheck -DFILE=%t.o --check-prefix=ERROR-RANGE %s
# ERROR-RANGE: error: [[FILE]]:(.text+0x0): relocation R_LARCH_CALL36 out of range: 137438953472 is not in [-137438953472, 137438953471]; references 'foo'

## Impossible case in reality becasue all LoongArch instructions are fixed 4-bytes long.
# RUN: not ld.lld %t.o --section-start=.text=0x20000 --section-start=.sec.foo=0x40001 -o /dev/null 2>&1 | \
# RUN:   FileCheck -DFILE=%t.o --check-prefix=ERROR-ALIGN %s
# ERROR-ALIGN: error: [[FILE]]:(.text+0x0): improper alignment for relocation R_LARCH_CALL36: 0x20001 is not aligned to 4 bytes

.text
.global _start
_start:
1:
  pcaddu18i $ra, 0
  jirl $zero, $ra, 0
  .reloc 1b, R_LARCH_CALL36, foo

.section .sec.foo,"ax"
.global foo
foo:
  nop
  ret
