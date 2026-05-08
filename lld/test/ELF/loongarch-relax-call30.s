# REQUIRES: loongarch
## Relax R_LARCH_CALL30, which involves the macro instructions call30/tail30.

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=loongarch32 -mattr=+relax a.s -o a.32.o
# RUN: llvm-mc -filetype=obj -triple=loongarch32 -mattr=+relax b.s -o b.32.o
# RUN: ld.lld -shared -soname=b.so b.32.o -o b.32.so
# RUN: ld.lld -T lds a.32.o b.32.so -o 32
# RUN: llvm-objdump -td --no-show-raw-insn 32 | FileCheck %s --check-prefix=RELAX

## --no-relax disables relaxation.
# RUN: ld.lld -T lds a.32.o b.32.so --no-relax -o 32.norelax
# RUN: llvm-objdump -td --no-show-raw-insn 32.norelax | FileCheck %s --check-prefix=NORELAX

# RELAX:       {{0*}}00010000 g       .text  {{0*}}0000001c _start
# RELAX:       {{0*}}0001001c g       .text  {{0*}}00000000 _start_end
# RELAX:       {{0*}}00010808 g       .mid   {{0*}}00000000 mid_end
# RELAX:       {{0*}}10010010 g       .high  {{0*}}00000000 high_end

# RELAX-LABEL: <_start>:
## offset = 0x10018 - 0x10000 = 24
# RELAX-NEXT:      10000:  bl     24 <a>
# RELAX-NEXT:              b      20 <a>
# RELAX-NEXT:              nop
# RELAX-NEXT:              nop
## offset = .plt(0x10400)+32 - 0x10010 = 1040
# RELAX-NEXT:      10010:  bl     1040 <bar+0x10420>
# RELAX-NEXT:              b      1036 <bar+0x10420>
# RELAX-EMPTY:
# RELAX-NEXT: <a>:
# RELAX-NEXT:      10018:  ret
# RELAX-EMPTY:

# RELAX-LABEL: <.mid>:
## offset = 0x10000 - 0x10800 = -2048
# RELAX-NEXT:      10800:  bl     -2048 <_start>
# RELAX-NEXT:              b      -2052 <_start>
# RELAX-EMPTY:

# RELAX-LABEL: <.mid2>:
## offset = 0x10000 - 0x1010000 = -16777216
# RELAX-NEXT:    1010000:  bl     -16777216 <_start>
# RELAX-NEXT:              b      -16777220 <_start>
# RELAX-EMPTY:

# RELAX-LABEL: <.high>:
## offset = 0x10000 - 0x10010000 = -0x10000000, hi=-1024, lo18=0
# RELAX-NEXT:   10010000:  pcaddu12i $ra, -65536
# RELAX-NEXT:              jirl   $ra, $ra, 0
# RELAX-NEXT:              pcaddu12i $t0, -65537
# RELAX-NEXT:              jirl   $zero, $t0, 4088
# RELAX-EMPTY:


# NORELAX-LABEL: <_start>:
## offset = 0x10020 - 0x10000 = 0x20, hi=0, lo18=32
# NORELAX-NEXT:    10000:  pcaddu12i $ra, 0
# NORELAX-NEXT:            jirl   $ra, $ra, 32
## offset = 0x10020 - 0x10008 = 0x18, hi=0, lo18=24
# NORELAX-NEXT:    10008:  pcaddu12i $t0, 0
# NORELAX-NEXT:            jirl   $zero, $t0, 24
## offset = .plt(0x10400)+32 - 0x10010 = 0x410, hi=0, lo18=1040
# NORELAX-NEXT:    10010:  pcaddu12i $ra, 0
# NORELAX-NEXT:            jirl   $ra, $ra, 1040
## offset = .plt(0x10400)+32 - 0x10018 = 0x408, hi=0, lo18=1032
# NORELAX-NEXT:    10018:  pcaddu12i $t0, 0
# NORELAX-NEXT:            jirl   $zero, $t0, 1032
# NORELAX-EMPTY:
# NORELAX-NEXT: <a>:
# NORELAX-NEXT:      10020:  ret
# NORELAX-EMPTY:

# NORELAX-LABEL: <.mid>:
## offset = 0x10000 - 0x10800 = -0x800, hi=0, lo18=-2048
# NORELAX-NEXT:    10800:  pcaddu12i $ra, -1
# NORELAX-NEXT:            jirl   $ra, $ra, 2048
# NORELAX-NEXT:            pcaddu12i $t0, -1
# NORELAX-NEXT:            jirl   $zero, $t0, 2040
# NORELAX-EMPTY:

# NORELAX-LABEL: <.mid2>:
## offset = 0x10000 - 0x1010000 = -0x1000000, hi=-64, lo18=0
# NORELAX-NEXT:  1010000:  pcaddu12i $ra, -4096
# NORELAX-NEXT:            jirl   $ra, $ra, 0
# NORELAX-NEXT:            pcaddu12i $t0, -4097
# NORELAX-NEXT:            jirl   $zero, $t0, 4088
# NORELAX-EMPTY:

# NORELAX-LABEL: <.high>:
## offset = 0x10000 - 0x10010000 = -0x10000000, hi=-1024, lo18=0
# NORELAX-NEXT: 10010000:  pcaddu12i $ra, -65536
# NORELAX-NEXT:            jirl   $ra, $ra, 0
# NORELAX-NEXT:            pcaddu12i $t0, -65537
# NORELAX-NEXT:            jirl   $zero, $t0, 4088
# NORELAX-EMPTY:

#--- a.s
.global _start, _start_end
_start:
  call30 a          # relaxed. la32: bl
  tail30 $t0, a     # relaxed. la32: b
.balign 16
  call30 bar        # PLT call30 can be relaxed. la32: bl
  tail30 $t0, bar   # PLT tail30 can be relaxed. la32: bl

a:
  ret
.size _start, . - _start
_start_end:

.section .mid,"ax",@progbits
  call30 _start         # relaxed. la32: bl
  tail30 $t0, _start    # relaxed. la32: b

.section .mid2,"ax",@progbits
  call30 _start         # relaxed. la32: bl
  tail30 $t0, _start    # relaxed. la32: b

.section .high,"ax",@progbits
  call30 _start         # exceed range, not relaxed
  tail30 $t0, _start    # exceed range, not relaxed

#--- b.s
.globl bar
bar:
  ret

#--- lds
SECTIONS {
  .text 0x10000 : { *(.text) }
  .plt 0x10400 : { *(.plt) }
  .mid 0x10800 : { *(.mid); mid_end = .; }
  .mid2 0x1010000 : { *(.mid2) }
  .high 0x10010000 : { *(.high); high_end = .; }
}
