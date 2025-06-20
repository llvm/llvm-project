# REQUIRES: loongarch

# RUN: llvm-mc --filetype=obj --triple=loongarch32 -mattr=+relax %s -o %t.32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 -mattr=+relax %s -o %t.64.o

# RUN: ld.lld --section-start=.text=0x10000 --section-start=.data=0x14000 %t.32.o -o %t.32
# RUN: ld.lld --section-start=.text=0x10000 --section-start=.data=0x14000 %t.64.o -o %t.64
# RUN: llvm-objdump -td --no-show-raw-insn %t.32 | FileCheck --check-prefixes=RELAX %s
# RUN: llvm-objdump -td --no-show-raw-insn %t.64 | FileCheck --check-prefixes=RELAX %s

# RUN: ld.lld --section-start=.text=0x10000 --section-start=.data=0x14000 %t.32.o -shared -o %t.32s
# RUN: ld.lld --section-start=.text=0x10000 --section-start=.data=0x14000 %t.64.o -shared -o %t.64s
# RUN: llvm-objdump -td --no-show-raw-insn %t.32s | FileCheck --check-prefixes=RELAX %s
# RUN: llvm-objdump -td --no-show-raw-insn %t.64s | FileCheck --check-prefixes=RELAX %s

# RUN: ld.lld --section-start=.text=0x10000 --section-start=.data=0x410000 %t.32.o -o %t.32o
# RUN: ld.lld --section-start=.text=0x10000 --section-start=.data=0x410000 %t.64.o -o %t.64o
# RUN: llvm-objdump -td --no-show-raw-insn %t.32o | FileCheck --check-prefixes=NORELAX32 %s
# RUN: llvm-objdump -td --no-show-raw-insn %t.64o | FileCheck --check-prefixes=NORELAX64 %s

# RELAX-LABEL: <_start>:
## offset = 0x14000 - 0x10000 = 4096<<2
# RELAX-NEXT:      10000:  pcaddi $a0, 4096
# RELAX-NEXT:              pcaddi $a0, 4095
# RELAX-NEXT:              pcaddi $a0, 4094
# RELAX-NEXT:              pcaddi $a0, 4093

# NORELAX32-LABEL: <_start>:
## offset exceed range of pcaddi
## offset = 0x410000 - 0x10000: 0x400 pages, page offset 0
# NORELAX32-NEXT:  10000:  pcalau12i     $a0, 1024
# NORELAX32-NEXT:          addi.w        $a0, $a0, 0
# NORELAX32-NEXT:          pcalau12i     $a0, 1024
# NORELAX32-NEXT:          ld.w          $a0, $a0, 4
# NORELAX32-NEXT:          pcalau12i     $a0, 1024
# NORELAX32-NEXT:          addi.w        $a0, $a0, 0
# NORELAX32-NEXT:          pcalau12i     $a0, 1024
# NORELAX32-NEXT:          ld.w          $a0, $a0, 4

# NORELAX64-LABEL: <_start>:
## offset exceed range of pcaddi
## offset = 0x410000 - 0x10000: 0x400 pages, page offset 0
# NORELAX64-NEXT:  10000:  pcalau12i     $a0, 1024
# NORELAX64-NEXT:          addi.d        $a0, $a0, 0
# NORELAX64-NEXT:          pcalau12i     $a0, 1024
# NORELAX64-NEXT:          ld.d          $a0, $a0, 8
# NORELAX64-NEXT:          pcalau12i     $a0, 1024
# NORELAX64-NEXT:          addi.d        $a0, $a0, 0
# NORELAX64-NEXT:          pcalau12i     $a0, 1024
# NORELAX64-NEXT:          ld.d          $a0, $a0, 8

.section .text
.global _start
_start:
  la.local  $a0, sym
  la.global $a0, sym
  la.pcrel  $a0, sym
  la.got    $a0, sym

.section .data
sym:
  .zero 4
