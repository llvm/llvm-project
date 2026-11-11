## Preserve fallthroughs for nonzero cross-fragment edges. The first two
## branches share a stub; the third needs a separate stub for its CFI state.

# RUN: llvm-mc -triple riscv64 -mattr=+c -filetype=obj %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: ld.lld --emit-relocs -e shared %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --data=%t.fdata --reorder-blocks=ext-tsp \
# RUN:   --split-functions --split-strategy=random2 --bolt-seed=4 \
# RUN:   --split-all-cold --pad-funcs-before=shared:2097152 --print-longjmp \
# RUN:   --print-only=shared | FileCheck %s --check-prefix=PROFILE
# RUN: llvm-objdump -d --disassemble-symbols=shared %t.bolt > %t.check
# RUN: llvm-dwarfdump --debug-frame %t.bolt >> %t.check
# RUN: FileCheck %s < %t.check

# CHECK-LABEL: <shared>:
# CHECK: beqz a0, 0x[[#%x,STUB:]]
# CHECK-NEXT: addi a3, a3, 0x1
# CHECK-NEXT: beqz a1, 0x[[#STUB]]
# CHECK-NEXT: mv a2, sp
# CHECK-NEXT: beqz a4, 0x[[#%x,CFI_STUB:]]
# CHECK-NEXT: li a0, 0x1
# CHECK-NEXT: [[#%x,RET:]]: {{.*}}ret
# CHECK-NEXT: [[#STUB]]: {{.*}}auipc t1,
# CHECK: [[#CFI_STUB]]: {{.*}}auipc t1,
## The shared stub keeps the CFA from the hot return; the other stub changes it.
# CHECK: 0x[[#RET]]: CFA=X2
# CHECK-NEXT: 0x[[#CFI_STUB]]: CFA=X12

# PROFILE: Exec Count : 30
# PROFILE: Successors: {{.*}}(mispreds: 12, count: 30)

  .text
  .globl shared
  .type shared, @function
shared:
  .cfi_startproc
# FDATA: 0 [unknown] 0 1 shared 0 0 100
  addi a3, a3, 1
branch1:
  beqz a0, cold
# FDATA: 1 shared #branch1# 1 shared #cold# 5 10
# FDATA: 1 shared #branch1# 1 shared #next# 0 90
next:
  addi a3, a3, 1
branch2:
  beqz a1, cold
# FDATA: 1 shared #branch2# 1 shared #cold# 7 20
# FDATA: 1 shared #branch2# 1 shared #next2# 0 70
next2:
  mv a2, sp
  .cfi_def_cfa a2, 0
branch3:
  beqz a4, cold
# FDATA: 1 shared #branch3# 1 shared #cold# 9 30
# FDATA: 1 shared #branch3# 1 shared #hot# 0 40
hot:
  li a0, 1
  .cfi_def_cfa sp, 0
  ret
cold:
  li t1, 0
  ret
  .cfi_endproc
  .size shared, .-shared
  .reloc 0, R_RISCV_NONE
