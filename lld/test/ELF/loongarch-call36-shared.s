# REQUIRES: loongarch
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc --filetype=obj --triple=loongarch64-unknown-elf %t/a.s -o %t/a.o
# RUN: ld.lld %t/a.o -shared -T %t/a.t -o %t/a.so
# RUN: llvm-readelf -x .got.plt %t/a.so | FileCheck --check-prefix=GOTPLT %s
# RUN: llvm-objdump -d --no-show-raw-insn %t/a.so | FileCheck %s

## PLT should be present in this case.
# CHECK:       Disassembly of section .plt:
# CHECK:       <.plt>:
##             bar@plt:
# CHECK:       1234520:  pcaddu12i $t3, 64{{$}}
# CHECK-NEXT:            ld.d $t3, $t3, 536{{$}}
# CHECK-NEXT:            jirl $t1, $t3, 0
# CHECK-NEXT:            nop

# CHECK:      Disassembly of section .text:
# CHECK:      <foo>:
## hi20 = bar@plt - pc + (1 << 17) >> 18 = 0x1234520 - 0x1274670 + 0x20000 >> 18 = -1
## lo18 = bar@plt - pc & (1 << 18) - 1 = 0x1234520 - 0x1274670 & 0x3ffff = -336
# CHECK-NEXT: pcaddu18i $t0, -1{{$}}
# CHECK-NEXT: jirl $zero, $t0, -336{{$}}

# GOTPLT:      section '.got.plt':
# GOTPLT-NEXT: 0x01274728 00000000 00000000 00000000 00000000
# GOTPLT-NEXT: 0x01274738 00452301 00000000

#--- a.t
SECTIONS {
 .plt   0x1234500: { *(.plt) }
 .text  0x1274670: { *(.text) }
}

#--- a.s
.text
.global foo
.global bar
foo:
    pcaddu18i $t0, 0
    jirl      $zero, $t0, 0
    .reloc foo, R_LARCH_CALL36, bar
