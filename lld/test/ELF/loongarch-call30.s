# REQUIRES: loongarch

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc --filetype=obj --triple=loongarch32-unknown-elf %t/a.s -o %t/a.o

# RUN: ld.lld %t/a.o --section-start=.text=0x20010 --section-start=.sec.foo=0x21020 -o %t/exe1
# RUN: llvm-objdump --no-show-raw-insn -d %t/exe1 | FileCheck --match-full-lines %s --check-prefix=EXE1
## hi20 = target - pc >> 12 = 0x21020 - 0x20010 >> 12 = 1
## lo12 = target - pc & (1 << 12) - 1 = 0x21020 - 0x20010 & 0xfff = 16
# EXE1:      20010: pcaddu12i $t0, 1
# EXE1-NEXT: 20014: jirl $zero, $t0, 16

# RUN: ld.lld %t/a.o --section-start=.text=0x20010 --section-start=.sec.foo=0x21820 -o %t/exe2
# RUN: llvm-objdump --no-show-raw-insn -d %t/exe2 | FileCheck --match-full-lines %s --check-prefix=EXE2
## hi20 = target - pc >> 12 = 0x21820 - 0x20010 >> 12 = 1
## lo12 = target - pc & (1 << 12) - 1 = 0x21820 - 0x20010 & 0xfff = 2064
# EXE2:      20010: pcaddu12i $t0, 1
# EXE2-NEXT: 20014: jirl $zero, $t0, 2064

# RUN: ld.lld %t/a.o -shared -T %t/a.t -o %t/a.so
# RUN: llvm-readelf -x .got.plt %t/a.so | FileCheck --check-prefix=GOTPLT %s
# RUN: llvm-objdump -d --no-show-raw-insn %t/a.so | FileCheck --check-prefix=SO %s
## PLT should be present in this case.
# SO:    Disassembly of section .plt:
# SO:    <.plt>:
##       foo@plt:
# SO:    1234520:  pcaddu12i $t3, 64{{$}}
# SO-NEXT:         ld.w $t3, $t3, 444{{$}}
# SO-NEXT:         jirl $t1, $t3, 0
# SO-NEXT:         nop

# SO:   Disassembly of section .text:
# SO:   <_start>:
## hi20 = foo@plt - pc >> 12 = 0x1234520 - 0x1274670 >> 12 = -65
## lo18 = foo@plt - pc & (1 << 12) - 1 = 0x1234520 - 0x1274670 & 0xfff = 3760
# SO-NEXT: pcaddu12i $t0, -65{{$}}
# SO-NEXT: jirl $zero, $t0, 3760{{$}}

# GOTPLT:      section '.got.plt':
# GOTPLT-NEXT: 0x012746d4 00000000 00000000 00452301

## Impossible case in reality becasue all LoongArch instructions are fixed 4-bytes long.
# RUN: not ld.lld %t/a.o --section-start=.text=0x20000 --section-start=.sec.foo=0x40001 -o /dev/null 2>&1 | \
# RUN:   FileCheck -DFILE=%t/a.o --check-prefix=ERROR-ALIGN %s
# ERROR-ALIGN: error: [[FILE]]:(.text+0x0): improper alignment for relocation R_LARCH_CALL30: 0x20001 is not aligned to 4 bytes

#--- a.t
SECTIONS {
 .plt   0x1234500: { *(.plt) }
 .text  0x1274670: { *(.text) }
}

#--- a.s
.text
.global _start
_start:
  .reloc ., R_LARCH_CALL30, foo
  pcaddu12i $t0, 0
  jirl      $zero, $t0, 0

.section .sec.foo,"awx"
.global foo
foo:
  ret
