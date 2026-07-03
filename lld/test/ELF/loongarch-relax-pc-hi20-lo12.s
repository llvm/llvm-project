# REQUIRES: loongarch
# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc --filetype=obj --triple=loongarch32 -mattr=+32s,+relax a.s -o a.32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 -mattr=+relax a.s -o a.64.o

# RUN: ld.lld --section-start=.text=0x10000 --section-start=.data=0x14000 a.32.o -o a.32
# RUN: ld.lld --section-start=.text=0x10000 --section-start=.data=0x14000 a.64.o -o a.64
# RUN: llvm-objdump -td --no-show-raw-insn a.32 | FileCheck --check-prefixes=RELAX %s
# RUN: llvm-objdump -td --no-show-raw-insn a.64 | FileCheck --check-prefixes=RELAX %s

# RUN: ld.lld --section-start=.text=0x10000 --section-start=.data=0x14000 a.32.o -shared -o a.32s
# RUN: ld.lld --section-start=.text=0x10000 --section-start=.data=0x14000 a.64.o -shared -o a.64s
# RUN: llvm-objdump -td --no-show-raw-insn a.32s | FileCheck --check-prefixes=RELAX %s
# RUN: llvm-objdump -td --no-show-raw-insn a.64s | FileCheck --check-prefixes=RELAX %s

# RUN: ld.lld --section-start=.text=0x10000 --section-start=.data=0x410000 a.32.o -o a.32o
# RUN: ld.lld --section-start=.text=0x10000 --section-start=.data=0x410000 a.64.o -o a.64o
# RUN: llvm-objdump -td --no-show-raw-insn a.32o | FileCheck --check-prefixes=NORELAX32 %s
# RUN: llvm-objdump -td --no-show-raw-insn a.64o | FileCheck --check-prefixes=NORELAX64 %s

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
## Not relaxation, convertion to PCRel.
# NORELAX32-NEXT:          pcalau12i     $a0, 1024
# NORELAX32-NEXT:          addi.w        $a0, $a0, 0
# NORELAX32-NEXT:          pcalau12i     $a0, 1024
# NORELAX32-NEXT:          addi.w        $a0, $a0, 0
# NORELAX32-NEXT:          pcalau12i     $a0, 1024
# NORELAX32-NEXT:          addi.w        $a0, $a0, 0

# NORELAX64-LABEL: <_start>:
## offset exceed range of pcaddi
## offset = 0x410000 - 0x10000: 0x400 pages, page offset 0
# NORELAX64-NEXT:  10000:  pcalau12i     $a0, 1024
# NORELAX64-NEXT:          addi.d        $a0, $a0, 0
## Not relaxation, convertion to PCRel.
# NORELAX64-NEXT:          pcalau12i     $a0, 1024
# NORELAX64-NEXT:          addi.d        $a0, $a0, 0
# NORELAX64-NEXT:          pcalau12i     $a0, 1024
# NORELAX64-NEXT:          addi.d        $a0, $a0, 0
# NORELAX64-NEXT:          pcalau12i     $a0, 1024
# NORELAX64-NEXT:          addi.d        $a0, $a0, 0


## GOT references with non-zero addends. No relaxation.
# RUN: llvm-mc --filetype=obj --triple=loongarch32 -mattr=+32s,+relax nonzero.s -o nonzero.32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 -mattr=+relax nonzero.s -o nonzero.64.o
# RUN: ld.lld --section-start=.text=0x10000 --section-start=.data=0x14000 nonzero.32.o -o nonzero.32
# RUN: ld.lld --section-start=.text=0x10000 --section-start=.data=0x14000 nonzero.64.o -o nonzero.64
# RUN: llvm-objdump -td --no-show-raw-insn nonzero.32 | FileCheck --check-prefixes=NONZERO32 %s
# RUN: llvm-objdump -td --no-show-raw-insn nonzero.64 | FileCheck --check-prefixes=NONZERO64 %s

# NONZERO32-LABEL: <_start>:
# NONZERO32-NEXT:      10000:  pcalau12i $a0, 4
# NONZERO32-NEXT:              ld.w      $a0, $a0, 8

# NONZERO64-LABEL: <_start>:
# NONZERO64-NEXT:      10000:  pcalau12i $a0, 4
# NONZERO64-NEXT:              ld.d      $a0, $a0, 12


#--- a.s
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


#--- nonzero.s
.section .text
.global _start
_start:
  la.got    $a0, sym+4

.section .data
sym:
  .zero 4
