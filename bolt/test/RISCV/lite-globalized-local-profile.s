## Check that RISC-V instrumentation profiles using globalized local labels
## drive block reordering in lite mode. foo only has a PG profile, while bar
## only has a legacy profile. For .Lpriority, the PG name must take priority.
# RUN: llvm-mc -triple=riscv64 --save-temp-labels -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs -e foo %t.o -o %t.exe
# RUN: llvm-bolt %t.exe --data=%p/Inputs/globalized-local-profile.fdata \
# RUN:   --lite=1 --reorder-blocks=ext-tsp --print-cfg -o %t.bolt \
# RUN:   2>&1 | FileCheck %s --check-prefix=PROFILE
# RUN: llvm-objdump -d --no-show-raw-insn --disassemble-symbols=foo,bar %t.bolt \
# RUN:   | FileCheck %s --check-prefix=DISASM

# PROFILE-LABEL: Binary Function "PG.Lfoo/1(*2)"
# PROFILE: Exec Count  : 100
# PROFILE-LABEL: Binary Function "PG.Lbar/1(*2)"
# PROFILE: Exec Count  : 200
# PROFILE-LABEL: Binary Function "PG.Lpriority"
# PROFILE: Exec Count  : 300

## Both matched profiles favor the taken edge, so the hot block becomes the
## fallthrough and beqz is inverted in the optimized binary.
# DISASM-LABEL: <foo>:
# DISASM-NEXT: {{.*}} bnez a0,
# DISASM-NEXT: {{.*}} li a0, 0x2
# DISASM-NEXT: {{.*}} ret
# DISASM-LABEL: <bar>:
# DISASM-NEXT: {{.*}} bnez a0,
# DISASM-NEXT: {{.*}} li a0, 0x2
# DISASM-NEXT: {{.*}} ret

  .option norvc
  .text
  .globl foo, bar
  .type foo, @function
  .type .Lfoo, @function
  .local .Lfoo
foo:
.Lfoo:
  beqz a0, 1f
  li a0, 1
  ret
1:
  li a0, 2
  ret
  .size foo, .-foo
  .size .Lfoo, .-foo

  .type bar, @function
  .type .Lbar, @function
  .local .Lbar
bar:
.Lbar:
  beqz a0, 1f
  li a0, 1
  ret
1:
  li a0, 2
  ret
  .size bar, .-bar
  .size .Lbar, .-bar

  .globl .Lpriority
  .type .Lpriority, @function
.Lpriority:
  ret
  .size .Lpriority, .-.Lpriority

## Force relocation mode.
  .reloc 0, R_RISCV_NONE
