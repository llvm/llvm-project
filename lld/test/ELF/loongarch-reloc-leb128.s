# REQUIRES: loongarch
# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax a.s -o a.o
# RUN: llvm-readobj -r -x .gcc_except_table -x .debug_rnglists -x .debug_loclists a.o | FileCheck %s --check-prefix=REL
# RUN: ld.lld -shared --gc-sections a.o -o a.so
# RUN: llvm-readelf -x .gcc_except_table -x .debug_rnglists -x .debug_loclists a.so | FileCheck %s

# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax extraspace.s -o extraspace64.o
# RUN: not ld.lld -shared extraspace64.o 2>&1 | FileCheck %s --check-prefix=ERROR
# ERROR: error: extraspace{{.*}}.o:(.rodata+0x0): extra space for uleb128

#--- a.s
.cfi_startproc
.cfi_lsda 0x1b,.LLSDA0
.cfi_endproc

.globl _start
_start:
foo:
  nop

.section .text.w,"axR"
w1:
  call36 foo    # 4 bytes after relaxation
w2:

.section .text.x,"ax"
x1:
  call36 foo    # 4 bytes after relaxation
x2:

.section .gcc_except_table,"a"
.LLSDA0:
.reloc ., R_LARCH_ADD_ULEB128, w1+130
.reloc ., R_LARCH_SUB_ULEB128, w2-1  # non-zero addend for SUB
.byte 0x7b
.uleb128 w2-w1+120
.uleb128 w1-w2+137
.uleb128 w2-w1+16376
.uleb128 w1-w2+16393
.uleb128 w2-w1+2097144
.uleb128 w1-w2+2097161

.section .debug_rnglists
.reloc ., R_LARCH_ADD_ULEB128, w1+130
.reloc ., R_LARCH_SUB_ULEB128, w2-1  # non-zero addend for SUB
.byte 0x7b
.uleb128 w2-w1+120
.uleb128 w1-w2+137
.uleb128 w2-w1+16376
.uleb128 w1-w2+16393
.uleb128 w2-w1+2097144
.uleb128 w1-w2+2097161

.section .debug_loclists
.reloc ., R_LARCH_ADD_ULEB128, w2+3
.reloc ., R_LARCH_SUB_ULEB128, w1+4  # SUB with a non-zero addend
.byte 0
.uleb128 x2-x1                       # references discarded symbols

# REL:        Hex dump of section '.gcc_except_table':
# REL-NEXT:   0x00000000 7b800080 00808000 80800080 80800080 {
# REL-NEXT:   0x00000010 808000                              .
# REL:        Hex dump of section '.debug_rnglists':
# REL-NEXT:   0x00000000 7b800080 00808000 80800080 80800080 {
# REL-NEXT:   0x00000010 808000                              .
# REL:        Hex dump of section '.debug_loclists':
# REL-NEXT:   0x00000000 0000                                  .

# CHECK:      Hex dump of section '.gcc_except_table':
# CHECK-NEXT: 0x00000238 7afc0085 01fcff00 858001fc ffff0085 z
# CHECK-NEXT: 0x00000248 808001                              .
# CHECK:      Hex dump of section '.debug_rnglists':
# CHECK-NEXT: 0x00000000 7afc0085 01fcff00 858001fc ffff0085 z
# CHECK-NEXT: 0x00000010 808001                              .
# CHECK:      Hex dump of section '.debug_loclists':
# CHECK-NEXT: 0x00000000 0300                                .

#--- extraspace.s
.text
w1:
  la.pcrel $t0, w1
w2:

.rodata
.reloc ., R_LARCH_ADD_ULEB128, w2
.reloc ., R_LARCH_SUB_ULEB128, w1
.fill 10, 1, 0x80
.byte 1
