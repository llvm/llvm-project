## A long jump through t0/x5 carries a RAS-pop hint even if t0 is dead.
## Use another dead register, or cancel splitting when only t0 is available.
## Keep the fragments beyond JAL range so the JALR cannot be relaxed away.

# RUN: llvm-mc -triple riscv64 -filetype=obj --defsym OTHER_SCRATCH=1 %s -o %t.o
# RUN: ld.lld --emit-relocs -e _start %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=all \
# RUN:   --pad-funcs-before=_start:2097152
# RUN: llvm-objdump -d --disassemble-symbols=_start %t.bolt | FileCheck %s

# RUN: llvm-mc -triple riscv64 -filetype=obj %s -o %t.only-t0.o
# RUN: ld.lld --emit-relocs -e _start %t.only-t0.o -o %t.only-t0.exe
# RUN: llvm-bolt %t.only-t0.exe -o %t.only-t0.bolt --split-functions \
# RUN:   --split-strategy=all 2>&1 | FileCheck %s --check-prefix=FALLBACK
# RUN: llvm-readelf -S %t.only-t0.bolt | FileCheck %s --check-prefix=SECTIONS
# RUN: llvm-objdump -d --disassemble-symbols=_start %t.only-t0.bolt | \
# RUN:   FileCheck %s --check-prefix=UNSPLIT --implicit-check-not=auipc

# CHECK-LABEL: <_start>:
# CHECK: auipc t1,
# CHECK-NEXT: jr {{.*}}(t1)
# CHECK: auipc t1,
# CHECK-NEXT: jr {{.*}}(t1)
# FALLBACK: BOLT-WARNING: keeping _start unsplit: no dead register for a RISC-V long jump
# FALLBACK: BOLT-INFO: RISC-V inserted long-jump stubs in 0 split functions; kept 1 functions together
# SECTIONS-NOT: .text.cold
# UNSPLIT-LABEL: <_start>:
# UNSPLIT: li t0, 0x0
# UNSPLIT-NEXT: ret
# UNSPLIT: li t0, 0x1
# UNSPLIT-NEXT: ret

  .text
  .globl _start
  .type _start, @function
_start:
  beqz a0, .Lcold
  li t0, 0
  .ifdef OTHER_SCRATCH
  li t1, 0
  .endif
  ret
.Lcold:
  li t0, 1
  .ifdef OTHER_SCRATCH
  li t1, 1
  .endif
  ret
  .size _start, .-_start

  .reloc 0, R_RISCV_NONE
