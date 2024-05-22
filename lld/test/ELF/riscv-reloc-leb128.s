# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax a.s -o a.o
# RUN: llvm-readobj -r -x .debug_rnglists -x .debug_loclists a.o | FileCheck %s --check-prefix=REL
# RUN: ld.lld -shared --gc-sections a.o -o a.so
# RUN: llvm-readelf -x .debug_rnglists -x .debug_loclists a.so | FileCheck %s

# REL:      .rela.debug_rnglists {
# REL-NEXT:   0x0 R_RISCV_SET_ULEB128 w1 0x83
# REL-NEXT:   0x0 R_RISCV_SUB_ULEB128 w2 0x0
# REL-NEXT:   0x1 R_RISCV_SET_ULEB128 w2 0x78
# REL-NEXT:   0x1 R_RISCV_SUB_ULEB128 w1 0x0
# REL-NEXT:   0x3 R_RISCV_SET_ULEB128 w1 0x89
# REL-NEXT:   0x3 R_RISCV_SUB_ULEB128 w2 0x0
# REL-NEXT:   0x5 R_RISCV_SET_ULEB128 w2 0x3FF8
# REL-NEXT:   0x5 R_RISCV_SUB_ULEB128 w1 0x0
# REL-NEXT:   0x8 R_RISCV_SET_ULEB128 w1 0x4009
# REL-NEXT:   0x8 R_RISCV_SUB_ULEB128 w2 0x0
# REL-NEXT:   0xB R_RISCV_SET_ULEB128 w2 0x1FFFF8
# REL-NEXT:   0xB R_RISCV_SUB_ULEB128 w1 0x0
# REL-NEXT:   0xF R_RISCV_SET_ULEB128 w1 0x200009
# REL-NEXT:   0xF R_RISCV_SUB_ULEB128 w2 0x0
# REL-NEXT: }
# REL:      .rela.debug_loclists {
# REL-NEXT:   0x0 R_RISCV_SET_ULEB128 w2 0x3
# REL-NEXT:   0x0 R_RISCV_SUB_ULEB128 w1 0x4
# REL-NEXT:   0x1 R_RISCV_SET_ULEB128 x2 0x0
# REL-NEXT:   0x1 R_RISCV_SUB_ULEB128 x1 0x0
# REL-NEXT: }

# REL:        Hex dump of section '.debug_rnglists':
# REL-NEXT:   0x00000000 7b800181 01808001 81800180 80800181 {
# REL-NEXT:   0x00000010 808001                              .
# REL:        Hex dump of section '.debug_loclists':
# REL-NEXT:   0x00000000 0008                                  .

# CHECK:      Hex dump of section '.debug_rnglists':
# CHECK-NEXT: 0x00000000 7ffc0085 01fcff00 858001fc ffff0085 .
# CHECK-NEXT: 0x00000010 808001                              .
# CHECK:      Hex dump of section '.debug_loclists':
# CHECK-NEXT: 0x00000000 0300                                .

# RUN: ld.lld -shared --gc-sections -z dead-reloc-in-nonalloc=.debug_loclists=0x7f a.o -o a127.so
# RUN: llvm-readelf -x .debug_loclists a127.so | FileCheck %s --check-prefix=CHECK127
# CHECK127:      Hex dump of section '.debug_loclists':
# CHECK127-NEXT: 0x00000000 037f                                .

# RUN: not ld.lld -shared --gc-sections -z dead-reloc-in-nonalloc=.debug_loclists=0x80 a.o 2>&1 | FileCheck %s --check-prefix=CHECK128
# CHECK128: error: a.o:(.debug_loclists+0x1): ULEB128 value 128 exceeds available space; references 'x2'

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax sub.s -o sub.o
# RUN: not ld.lld -shared sub.o 2>&1 | FileCheck %s --check-prefix=SUB
# SUB: error: sub.o:(.debug_rnglists+0x8): unknown relocation (61) against symbol w2

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax unpaired1.s -o unpaired1.o
# RUN: not ld.lld -shared unpaired1.o 2>&1 | FileCheck %s --check-prefix=UNPAIRED
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax unpaired2.s -o unpaired2.o
# RUN: not ld.lld -shared unpaired2.o 2>&1 | FileCheck %s --check-prefix=UNPAIRED
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax unpaired3.s -o unpaired3.o
# RUN: not ld.lld -shared unpaired3.o 2>&1 | FileCheck %s --check-prefix=UNPAIRED
# UNPAIRED: error: {{.*}}.o:(.debug_rnglists+0x8): R_RISCV_SET_ULEB128 not paired with R_RISCV_SUB_SET128

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax overflow.s -o overflow.o
# RUN: not ld.lld -shared overflow.o 2>&1 | FileCheck %s --check-prefix=OVERFLOW
# OVERFLOW: error: overflow.o:(.debug_rnglists+0x8): ULEB128 value 128 exceeds available space; references 'w2'

#--- a.s
.section .text.w,"axR"
w1:
  call foo    # 4 bytes after relaxation
w2:

.section .text.x,"ax"
x1:
  call foo    # 4 bytes after relaxation
x2:

.section .debug_rnglists
.uleb128 w1-w2+131                   # initial value: 0x7b
.uleb128 w2-w1+120                   # initial value: 0x0180
.uleb128 w1-w2+137                   # initial value: 0x0181
.uleb128 w2-w1+16376                 # initial value: 0x018080
.uleb128 w1-w2+16393                 # initial value: 0x018081
.uleb128 w2-w1+2097144               # initial value: 0x01808080
.uleb128 w1-w2+2097161               # initial value: 0x01808081

.section .debug_loclists
.reloc ., R_RISCV_SET_ULEB128, w2+3
.reloc ., R_RISCV_SUB_ULEB128, w1+4  # SUB with a non-zero addend
.byte 0
.uleb128 x2-x1                       # references discarded symbols

#--- sub.s
w1: call foo; w2:
.section .debug_rnglists
.quad 0;
.reloc ., R_RISCV_SUB_ULEB128, w2+120
.byte 0x7f

#--- unpaired1.s
w1: call foo; w2:
.section .debug_rnglists
.quad 0;
.reloc ., R_RISCV_SET_ULEB128, w2+120
.byte 0x7f

#--- unpaired2.s
w1: call foo; w2:
.section .debug_rnglists
.quad 0
.reloc ., R_RISCV_SET_ULEB128, w2+120
.reloc .+1, R_RISCV_SUB_ULEB128, w1
.byte 0x7f

#--- unpaired3.s
w1: call foo; w2:
.section .debug_rnglists
.quad 0
.reloc ., R_RISCV_SET_ULEB128, w2+120
.reloc ., R_RISCV_SUB64, w1
.byte 0x7f

#--- overflow.s
w1: call foo; w2:
.section .debug_rnglists
.quad 0
.reloc ., R_RISCV_SET_ULEB128, w2+124
.reloc ., R_RISCV_SUB_ULEB128, w1
.byte 0x7f
