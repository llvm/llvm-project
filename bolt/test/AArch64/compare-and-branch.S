# This test checks that splitting functions which contain short range
# conditional branches works in compact code model without relying on
# relocations. Also checks that splitting works in non-relocation mode,
# in order to test the branch inversion on those instructions.

# REQUIRES: system-linux, asserts

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t -DRELOCATION_MODE=1
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions --compact-code-model
# RUN: llvm-objdump -r %t | FileCheck %s --check-prefix=CHECK-RELOCS
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=RELOC-MODE

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t -DRELOCATION_MODE=0
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions
# RUN: llvm-objdump -r %t | FileCheck %s --check-prefix=CHECK-NO-RELOCS
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=NON-RELOC-MODE

  .globl immediate_increment
  .type immediate_increment, %function
immediate_increment:
# FDATA: 1 immediate_increment #.entry_immediate_increment# 10
.entry_immediate_increment:
## Test immediate increment when inverting the branch.
    cbgt x0, #0, .Lcold_immediate_increment
    mov x0, #1
.Lcold_immediate_increment:
    ret

  .globl immediate_decrement
  .type immediate_decrement, %function
immediate_decrement:
# FDATA: 1 immediate_decrement #.entry_immediate_decrement# 10
.entry_immediate_decrement:
## Test immediate decrement when inverting the branch.
    cblo x0, #1, .Lcold_immediate_decrement
    mov x0, #2
.Lcold_immediate_decrement:
    ret

  .globl register_swap
  .type register_swap, %function
register_swap:
# FDATA: 1 register_swap #.entry_register_swap# 10
.entry_register_swap:
## Test register swap when inverting the branch.
    cbge x0, x1, .Lcold_register_swap
    mov x0, #3
.Lcold_register_swap:
    ret

  .globl irreversible
  .type irreversible, %function
irreversible:
# FDATA: 1 irreversible #.entry_irreversible# 10
.entry_irreversible:
    cbgt x0, #63, .Lcold_irreversible
    mov x0, #4
.Lcold_irreversible:
    ret

## Force relocation mode.
.if RELOCATION_MODE
.reloc 0, R_AARCH64_NONE
.endif


# CHECK-RELOCS: R_AARCH64_NONE *ABS*
# CHECK-RELOCS-NOT: R_AARCH64_


# CHECK-NO-RELOCS-NOT: R_AARCH64_


# RELOC-MODE: Disassembly of section .text:

# RELOC-MODE: <immediate_increment>:
# RELOC-MODE-NEXT:            {{.*}} cbgt x0, #0x0, 0x[[ADDR0:[0-9a-f]+]] <{{.*}}>
# RELOC-MODE-NEXT:            {{.*}} b              0x[[ADDR1:[0-9a-f]+]] <{{.*}}>
# RELOC-MODE-NEXT: [[ADDR0]]: {{.*}} b              0x[[ADDR2:[0-9a-f]+]] <{{.*}}>

# RELOC-MODE: <immediate_decrement>:
# RELOC-MODE-NEXT:            {{.*}} cblo x0, #0x1, 0x[[ADDR3:[0-9a-f]+]] <{{.*}}>
# RELOC-MODE-NEXT:            {{.*}} b              0x[[ADDR4:[0-9a-f]+]] <{{.*}}>
# RELOC-MODE-NEXT: [[ADDR3]]: {{.*}} b              0x[[ADDR5:[0-9a-f]+]] <{{.*}}>

# RELOC-MODE: <register_swap>:
# RELOC-MODE-NEXT:            {{.*}} cbge x0, x1, 0x[[ADDR6:[0-9a-f]+]] <{{.*}}>
# RELOC-MODE-NEXT:            {{.*}} b            0x[[ADDR7:[0-9a-f]+]] <{{.*}}>
# RELOC-MODE-NEXT: [[ADDR6]]: {{.*}} b            0x[[ADDR8:[0-9a-f]+]] <{{.*}}>

# RELOC-MODE: <irreversible>:
# RELOC-MODE-NEXT:            {{.*}} cbgt x0, #0x3f, 0x[[ADDR9:[0-9a-f]+]] <{{.*}}>
# RELOC-MODE-NEXT:            {{.*}} b               0x[[ADDR10:[0-9a-f]+]] <{{.*}}>
# RELOC-MODE-NEXT: [[ADDR9]]: {{.*}} b               0x[[ADDR11:[0-9a-f]+]] <{{.*}}>


# RELOC-MODE: Disassembly of section .text.cold:

# RELOC-MODE: <immediate_increment.cold.0>:
# RELOC-MODE-NEXT: [[ADDR1]]: {{.*}} mov x0, #0x1 // =1
# RELOC-MODE-NEXT: [[ADDR2]]: {{.*}} ret

# RELOC-MODE: <immediate_decrement.cold.0>:
# RELOC-MODE-NEXT: [[ADDR4]]: {{.*}} mov x0, #0x2 // =2
# RELOC-MODE-NEXT: [[ADDR5]]: {{.*}} ret

# RELOC-MODE: <register_swap.cold.0>:
# RELOC-MODE-NEXT: [[ADDR7]]: {{.*}} mov x0, #0x3 // =3
# RELOC-MODE-NEXT: [[ADDR8]]: {{.*}} ret

# RELOC-MODE: <irreversible.cold.0>:
# RELOC-MODE-NEXT: [[ADDR10]]: {{.*}} mov x0, #0x4 // =4
# RELOC-MODE-NEXT: [[ADDR11]]: {{.*}} ret


# NON-RELOC-MODE: Disassembly of section .text:

# NON-RELOC-MODE: <immediate_increment>:
# NON-RELOC-MODE-NEXT:            {{.*}} cblt x0, #0x1, 0x[[ADDR0:[0-9a-f]+]] <{{.*}}>
# NON-RELOC-MODE-NEXT:            {{.*}} b              0x[[ADDR1:[0-9a-f]+]] <{{.*}}>
# NON-RELOC-MODE-NEXT: [[ADDR0]]: {{.*}} b              0x[[ADDR2:[0-9a-f]+]] <{{.*}}>

# NON-RELOC-MODE: <immediate_decrement>:
# NON-RELOC-MODE-NEXT:            {{.*}} cbhi x0, #0x0, 0x[[ADDR3:[0-9a-f]+]] <{{.*}}>
# NON-RELOC-MODE-NEXT:            {{.*}} b              0x[[ADDR4:[0-9a-f]+]] <{{.*}}>
# NON-RELOC-MODE-NEXT: [[ADDR3]]: {{.*}} b              0x[[ADDR5:[0-9a-f]+]] <{{.*}}>

# NON-RELOC-MODE: <register_swap>:
# NON-RELOC-MODE-NEXT:            {{.*}} cbgt x1, x0, 0x[[ADDR6:[0-9a-f]+]] <{{.*}}>
# NON-RELOC-MODE-NEXT:            {{.*}} b            0x[[ADDR7:[0-9a-f]+]] <{{.*}}>
# NON-RELOC-MODE-NEXT: [[ADDR6]]: {{.*}} b            0x[[ADDR8:[0-9a-f]+]] <{{.*}}>

# NON-RELOC-MODE: <irreversible>:
# NON-RELOC-MODE-NEXT:            {{.*}} cbgt x0, #0x3f, 0x[[ADDR9:[0-9a-f]+]] <{{.*}}>
# NON-RELOC-MODE-NEXT:            {{.*}} b               0x[[ADDR10:[0-9a-f]+]] <{{.*}}>
# NON-RELOC-MODE-NEXT: [[ADDR9]]: {{.*}} b               0x[[ADDR11:[0-9a-f]+]] <{{.*}}>


# NON-RELOC-MODE: Disassembly of section .bolt.text:

# NON-RELOC-MODE: <immediate_increment.cold.0>:
# NON-RELOC-MODE-NEXT: [[ADDR2]]: {{.*}} mov x0, #0x1 // =1
# NON-RELOC-MODE-NEXT: [[ADDR1]]: {{.*}} ret

# NON-RELOC-MODE: <immediate_decrement.cold.0>:
# NON-RELOC-MODE-NEXT: [[ADDR5]]: {{.*}} mov x0, #0x2 // =2
# NON-RELOC-MODE-NEXT: [[ADDR4]]: {{.*}} ret

# NON-RELOC-MODE: <register_swap.cold.0>:
# NON-RELOC-MODE-NEXT: [[ADDR8]]: {{.*}} mov x0, #0x3 // =3
# NON-RELOC-MODE-NEXT: [[ADDR7]]: {{.*}} ret

# NON-RELOC-MODE: <irreversible.cold.0>:
# NON-RELOC-MODE-NEXT: [[ADDR10]]: {{.*}} mov x0, #0x4 // =4
# NON-RELOC-MODE-NEXT: [[ADDR11]]: {{.*}} ret
