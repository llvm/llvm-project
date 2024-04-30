# REQUIRES: loongarch
# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax a.s -o a.o
# RUN: llvm-readobj -r -x .gcc_except_table -x .debug_rnglists -x .debug_loclists a.o | FileCheck %s --check-prefix=REL
# RUN: ld.lld -shared --gc-sections a.o -o a.so
# RUN: llvm-readelf -x .gcc_except_table -x .debug_rnglists -x .debug_loclists a.so | FileCheck %s

# RUN: llvm-mc --filetype=obj --triple=loongarch32 --mattr=+relax a.s -o a32.o
# RUN: llvm-readobj -r -x .gcc_except_table -x .debug_rnglists -x .debug_loclists a32.o | FileCheck %s --check-prefix=REL
# RUN: ld.lld -shared --gc-sections a32.o -o a32.so
# RUN: llvm-readelf -x .gcc_except_table -x .debug_rnglists -x .debug_loclists a32.so | FileCheck %s

# RUN: llvm-mc --filetype=obj --triple=loongarch32 --mattr=+relax extraspace.s -o extraspace32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax extraspace.s -o extraspace64.o
# RUN: not ld.lld -shared extraspace32.o 2>&1 | FileCheck %s --check-prefix=ERROR
# RUN: not ld.lld -shared extraspace64.o 2>&1 | FileCheck %s --check-prefix=ERROR
# ERROR: error: extraspace{{.*}}.o:(.rodata+0x0): extra space for uleb128

#--- a.s
.cfi_startproc
.cfi_lsda 0x1b,.LLSDA0
.cfi_endproc

.section .text.w,"axR"
break 0; break 0; break 0; w1:
  .p2align 4    # 4 bytes after relaxation
w2: break 0

.section .text.x,"ax"
break 0; break 0; break 0; x1:
  .p2align 4    # 4 bytes after relaxation
x2: break 0

.section .gcc_except_table,"a"
.LLSDA0:
.uleb128 w2-w1+116                   # initial value: 0x0080
.uleb128 w1-w2+141                   # initial value: 0x0080
.uleb128 w2-w1+16372                 # initial value: 0x008080
.uleb128 w1-w2+16397                 # initial value: 0x008080
.uleb128 w2-w1+2097140               # initial value: 0x00808080
.uleb128 w1-w2+2097165               # initial value: 0x00808080

.section .debug_rnglists
.uleb128 w2-w1+116                   # initial value: 0x0080
.uleb128 w1-w2+141                   # initial value: 0x0080
.uleb128 w2-w1+16372                 # initial value: 0x008080
.uleb128 w1-w2+16397                 # initial value: 0x008080
.uleb128 w2-w1+2097140               # initial value: 0x00808080
.uleb128 w1-w2+2097165               # initial value: 0x00808080

.section .debug_loclists
.uleb128 x2-x1                       # references discarded symbols

# REL:      Section ({{.*}}) .rela.debug_rnglists {
# REL-NEXT:   0x0 R_LARCH_ADD_ULEB128 w2 0x74
# REL-NEXT:   0x0 R_LARCH_SUB_ULEB128 w1 0x0
# REL-NEXT:   0x2 R_LARCH_ADD_ULEB128 w1 0x8D
# REL-NEXT:   0x2 R_LARCH_SUB_ULEB128 w2 0x0
# REL-NEXT:   0x4 R_LARCH_ADD_ULEB128 w2 0x3FF4
# REL-NEXT:   0x4 R_LARCH_SUB_ULEB128 w1 0x0
# REL-NEXT:   0x7 R_LARCH_ADD_ULEB128 w1 0x400D
# REL-NEXT:   0x7 R_LARCH_SUB_ULEB128 w2 0x0
# REL-NEXT:   0xA R_LARCH_ADD_ULEB128 w2 0x1FFFF4
# REL-NEXT:   0xA R_LARCH_SUB_ULEB128 w1 0x0
# REL-NEXT:   0xE R_LARCH_ADD_ULEB128 w1 0x20000D
# REL-NEXT:   0xE R_LARCH_SUB_ULEB128 w2 0x0
# REL-NEXT: }
# REL:      Section ({{.*}}) .rela.debug_loclists {
# REL-NEXT:   0x0 R_LARCH_ADD_ULEB128 x2 0x0
# REL-NEXT:   0x0 R_LARCH_SUB_ULEB128 x1 0x0
# REL-NEXT: }

# REL:      Hex dump of section '.gcc_except_table':
# REL-NEXT: 0x00000000 80008000 80800080 80008080 80008080 .
# REL-NEXT: 0x00000010 8000                                .
# REL:      Hex dump of section '.debug_rnglists':
# REL-NEXT: 0x00000000 80008000 80800080 80008080 80008080 .
# REL-NEXT: 0x00000010 8000                                .
# REL:      Hex dump of section '.debug_loclists':
# REL-NEXT: 0x00000000 00                                  .

# CHECK:      Hex dump of section '.gcc_except_table':
# CHECK-NEXT: 0x[[#%x,]] f8008901 f8ff0089 8001f8ff ff008980 .
# CHECK-NEXT: 0x[[#%x,]] 8001                                .
# CHECK:      Hex dump of section '.debug_rnglists':
# CHECK-NEXT: 0x00000000 f8008901 f8ff0089 8001f8ff ff008980 .
# CHECK-NEXT: 0x00000010 8001                                .
# CHECK:      Hex dump of section '.debug_loclists':
# CHECK-NEXT: 0x00000000 00                                  .

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
