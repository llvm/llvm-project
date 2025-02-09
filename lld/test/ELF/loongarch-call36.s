# REQUIRES: loongarch

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc --filetype=obj --triple=loongarch64-unknown-elf %t/a.s -o %t/a.o

# RUN: ld.lld %t/a.o --section-start=.text=0x20010 --section-start=.sec.foo=0x60020 -o %t/exe1
# RUN: llvm-objdump --no-show-raw-insn -d %t/exe1 | FileCheck --match-full-lines %s --check-prefix=EXE1
## hi20 = target - pc + (1 << 17) >> 18 = 0x60020 - 0x20010 + 0x20000 >> 18 = 1
## lo18 = target - pc & (1 << 18) - 1 = 0x60020 - 0x20010 & 0x3ffff = 16
# EXE1:      20010: pcaddu18i $t0, 1
# EXE1-NEXT: 20014: jirl $zero, $t0, 16

# RUN: ld.lld %t/a.o --section-start=.text=0x20010 --section-start=.sec.foo=0x40020 -o %t/exe2
# RUN: llvm-objdump --no-show-raw-insn -d %t/exe2 | FileCheck --match-full-lines %s --check-prefix=EXE2
## hi20 = target - pc + (1 << 17) >> 18 = 0x40020 - 0x20010 + 0x20000 >> 18 = 1
## lo18 = target - pc & (1 << 18) - 1 = 0x40020 - 0x20010 & 0x3ffff = -131056
# EXE2:      20010: pcaddu18i $t0, 1
# EXE2-NEXT: 20014: jirl $zero, $t0, -131056

# RUN: ld.lld %t/a.o -shared -T %t/a.t -o %t/a.so
# RUN: llvm-readelf -x .got.plt %t/a.so | FileCheck --check-prefix=GOTPLT %s
# RUN: llvm-objdump -d --no-show-raw-insn %t/a.so | FileCheck --check-prefix=SO %s
## PLT should be present in this case.
# SO:    Disassembly of section .plt:
# SO:    <.plt>:
##       foo@plt:
# SO:    1234520:  pcaddu12i $t3, 64{{$}}
# SO-NEXT:         ld.d $t3, $t3, 544{{$}}
# SO-NEXT:         jirl $t1, $t3, 0
# SO-NEXT:         nop

# SO:   Disassembly of section .text:
# SO:   <_start>:
## hi20 = foo@plt - pc + (1 << 17) >> 18 = 0x1234520 - 0x1274670 + 0x20000 >> 18 = -1
## lo18 = foo@plt - pc & (1 << 18) - 1 = 0x1234520 - 0x1274670 & 0x3ffff = -336
# SO-NEXT: pcaddu18i $t0, -1{{$}}
# SO-NEXT: jirl $zero, $t0, -336{{$}}

# GOTPLT:      section '.got.plt':
# GOTPLT-NEXT: 0x01274730 00000000 00000000 00000000 00000000
# GOTPLT-NEXT: 0x01274740 00452301 00000000

# RUN: not ld.lld %t/a.o --section-start=.text=0x20000 --section-start=.sec.foo=0x2000020000 -o /dev/null 2>&1 | \
# RUN:   FileCheck -DFILE=%t/a.o --check-prefix=ERROR-RANGE %s
# ERROR-RANGE: error: [[FILE]]:(.text+0x0): relocation R_LARCH_CALL36 out of range: 137438953472 is not in [-137439084544, 137438822399]; references 'foo'

## Impossible case in reality becasue all LoongArch instructions are fixed 4-bytes long.
# RUN: not ld.lld %t/a.o --section-start=.text=0x20000 --section-start=.sec.foo=0x40001 -o /dev/null 2>&1 | \
# RUN:   FileCheck -DFILE=%t/a.o --check-prefix=ERROR-ALIGN %s
# ERROR-ALIGN: error: [[FILE]]:(.text+0x0): improper alignment for relocation R_LARCH_CALL36: 0x20001 is not aligned to 4 bytes

#--- a.t
SECTIONS {
 .plt   0x1234500: { *(.plt) }
 .text  0x1274670: { *(.text) }
}

#--- a.s
.text
.global _start
_start:
  .reloc ., R_LARCH_CALL36, foo
  pcaddu18i $t0, 0
  jirl      $zero, $t0, 0

.section .sec.foo,"awx"
.global foo
foo:
  ret
