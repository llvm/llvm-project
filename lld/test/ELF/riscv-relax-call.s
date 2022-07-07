# REQUIRES: riscv
## Relax R_RISCV_CALL and R_RISCV_CALL_PLT.

# RUN: rm -rf %t && split-file %s %t && cd %t

## Without RVC
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax a.s -o a.32.o
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax b.s -o b.32.o
# RUN: ld.lld -shared -soname=b.so b.32.o -o b.32.so
# RUN: ld.lld -T lds a.32.o b.32.so -o 32
# RUN: llvm-objdump -td --no-show-raw-insn -M no-aliases 32 | FileCheck %s --check-prefix=NORVC

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax a.s -o a.64.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax b.s -o b.64.o
# RUN: ld.lld -shared -soname=b.so b.64.o -o b.64.so
# RUN: ld.lld -T lds a.64.o b.64.so -o 64
# RUN: llvm-objdump -td --no-show-raw-insn -M no-aliases 64 | FileCheck %s --check-prefix=NORVC

## RVC
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c,+relax a.s -o a.32c.o
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c,+relax b.s -o b.32c.o
# RUN: ld.lld -shared -soname=b.so b.32c.o -o b.32c.so
# RUN: ld.lld -T lds a.32c.o b.32c.so -o 32c
# RUN: llvm-objdump -td --no-show-raw-insn -M no-aliases 32c | FileCheck %s --check-prefixes=RVC,RVC32

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax a.s -o a.64c.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax b.s -o b.64c.o
# RUN: ld.lld -shared -soname=b.so b.64c.o -o b.64c.so
# RUN: ld.lld -T lds a.64c.o b.64c.so -o 64c
# RUN: llvm-objdump -td --no-show-raw-insn -M no-aliases 64c | FileCheck %s --check-prefixes=RVC,RVC64

## --no-relax disables relaxation.
# RUN: ld.lld -T lds a.64.o b.64.so --no-relax -o 64.norelax
# RUN: llvm-objdump -td --no-show-raw-insn -M no-aliases 64.norelax | FileCheck %s --check-prefixes=NORELAX

# NORVC:       00010000 g       .text  {{0*}}0000001c _start
# NORVC:       0001001c g       .text  {{0*}}00000000 _start_end
# NORVC:       00010808 g       .mid   {{0*}}00000000 mid_end
# NORVC:       00110016 g       .high  {{0*}}00000000 high_end

# NORVC-LABEL: <_start>:
# NORVC-NEXT:    10000:  jal    zero, {{.*}} <a>
# NORVC-NEXT:            jal    zero, {{.*}} <a>
# NORVC-NEXT:            addi   zero, zero, 0
# NORVC-NEXT:            addi   zero, zero, 0
# NORVC-NEXT:    10010:  jal    ra, {{.*}} <a>
# NORVC-NEXT:            jal    ra, 0x10420
# NORVC-EMPTY:

# NORVC-LABEL: <.mid>:
# NORVC-NEXT:    10800:  jal    ra, {{.*}} <_start>
# NORVC-NEXT:            jal    ra, {{.*}} <_start>
# NORVC-EMPTY:

# NORVC-LABEL: <.mid2>:
# NORVC-NEXT:    1080c:  jal    ra, {{.*}} <_start>
# NORVC-EMPTY:

# NORVC-LABEL: <.high>:
# NORVC-NEXT:   110006:  auipc  ra, 1048320
# NORVC-NEXT:            jalr   ra, -6(ra)
# NORVC-NEXT:            auipc  ra, 1048320
# NORVC-NEXT:            jalr   ra, -14(ra)
# NORVC-EMPTY:

# RVC32:       00010000 g       .text  00000016 _start
# RVC32:       00010016 g       .text  00000000 _start_end
# RVC32:       00010806 g       .mid   00000000 mid_end
# RVC32:       0011000c g       .high  00000000 high_end
# RVC64:       0000000000010000 g       .text  000000000000001a _start
# RVC64:       000000000001001a g       .text  0000000000000000 _start_end
# RVC64:       0000000000010808 g       .mid   0000000000000000 mid_end
# RVC64:       0000000000110014 g       .high  0000000000000000 high_end

# RVC-LABEL:   <_start>:
# RVC-NEXT:      10000:  c.j    {{.*}} <a>
# RVC-NEXT:              c.j    {{.*}} <a>
# RVC-NEXT:              addi   zero, zero, 0
# RVC-NEXT:              addi   zero, zero, 0
# RVC-NEXT:              addi   zero, zero, 0
# RVC32-NEXT:    10010:  c.jal  {{.*}} <a>
# RVC32-NEXT:            c.jal  0x10420
# RVC64-NEXT:    10010:  jal    ra, {{.*}} <a>
# RVC64-NEXT:            jal    ra, 0x10420
# RVC-EMPTY:
# RVC-NEXT:    <a>:
# RVC-NEXT:              c.jr   ra
# RVC-EMPTY:

# RVC-LABEL:   <.mid>:
# RVC32-NEXT:    10800:  c.jal  {{.*}} <_start>
# RVC64-NEXT:    10800:  jal    ra, {{.*}} <_start>
# RVC-NEXT:              jal    ra, {{.*}} <_start>
# RVC-EMPTY:

# RVC-LABEL:   <.mid2>:
# RVC32-NEXT:    1080a:  jal    ra, {{.*}} <_start>
# RVC64-NEXT:    1080c:  jal    ra, {{.*}} <_start>
# RVC-EMPTY:

# RVC-LABEL:   <.high>:
# RVC32-NEXT:   110000:  jal    ra, 0x10000 <_start>
# RVC32-NEXT:            auipc  ra, 1048320
# RVC32-NEXT:            jalr   ra, -4(ra)
# RVC64-NEXT:   110004:  auipc  ra, 1048320
# RVC64-NEXT:            jalr   ra, -4(ra)
# RVC64-NEXT:            auipc  ra, 1048320
# RVC64-NEXT:            jalr   ra, -12(ra)
# RVC-EMPTY:

# NORELAX-LABEL: <_start>:
# NORELAX-NEXT:    10000:  auipc  t1, 0
# NORELAX-NEXT:            jalr   zero, 32(t1)
# NORELAX-NEXT:            auipc  t0, 0
# NORELAX-NEXT:            jalr   zero, 24(t0)
# NORELAX-NEXT:    10010:  auipc  ra, 0
# NORELAX-NEXT:            jalr   ra, 16(ra)
# NORELAX-NEXT:            auipc  ra, 0
# NORELAX-NEXT:            jalr   ra, 1032(ra)
# NORELAX-EMPTY:

#--- a.s
.global _start, _start_end
_start:
  tail a@plt
  jump a, t0
.balign 16
  call a          # rv32c: c.jal; rv64c: jal
  call bar        # PLT call can be relaxed. rv32c: c.jal; rv64c: jal

a:
  ret
.size _start, . - _start
_start_end:

.section .mid,"ax",@progbits
  call _start@plt # rv32c: c.jal; rv64c: jal
  call _start@plt

.section .mid2,"ax",@progbits
  call _start@plt

.section .high,"ax",@progbits
  call _start@plt # relaxable for %t/32c
  call _start@plt # not relaxed

#--- b.s
.globl bar
bar:
  ret

#--- lds
SECTIONS {
  .text 0x10000 : { *(.text) }
  .plt 0x10400 : { *(.plt) }
  .mid 0x10800 : { *(.mid); mid_end = .; }
  .mid2 mid_end+4 : { *(.mid2) }
  # 22 is the size of _start in %t/32c (RVC32).
  .high 0x110000+(_start_end-_start)-22 : { *(.high); high_end = .; }
}
