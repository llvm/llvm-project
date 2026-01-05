# This test checks that when spliting functions which contain short range
# conditional branches, we choose such a code layout that relocations are
# not needed.

# REQUIRES: system-linux, asserts

# RUN: %clang %cflags -Wl,-q %s -o %t
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

  .text

  .globl  foo
  .type foo, %function
foo:
.entry_foo:
# FDATA: 1 foo #.entry_foo# 10
    cbz x0, .Lcold_foo
    ret
.Lcold_foo:
    mov x0, #1
    ret

  .globl  bar
  .type bar, %function
bar:
.entry_bar:
# FDATA: 1 bar  #.entry_bar# 10
    tbz x0, #1, .Lcold_bar
    ret
.Lcold_bar:
    mov x0, #2
    ret

## Force relocation mode.
.reloc 0, R_AARCH64_NONE


# CHECK: Disassembly of section .text:

# CHECK: <foo>:
# CHECK-NEXT:            {{.*}} cbnz x0, 0x[[ADDR0:[0-9a-f]+]] <{{.*}}>
# CHECK-NEXT:            {{.*}} b        0x[[ADDR1:[0-9a-f]+]] <{{.*}}>
# CHECK-NEXT: [[ADDR0]]: {{.*}} b        0x[[ADDR2:[0-9a-f]+]] <{{.*}}>

# CHECK: <bar>:
# CHECK-NEXT:            {{.*}} tbnz w0, #0x1, 0x[[ADDR3:[0-9a-f]+]] <{{.*}}>
# CHECK-NEXT:            {{.*}} b              0x[[ADDR4:[0-9a-f]+]] <{{.*}}>
# CHECK-NEXT: [[ADDR3]]: {{.*}} b              0x[[ADDR5:[0-9a-f]+]] <{{.*}}>

# CHECK: Disassembly of section .text.cold:

# CHECK: <foo.cold.0>:
# CHECK-NEXT: [[ADDR2]]: {{.*}} ret
# CHECK-NEXT: [[ADDR1]]: {{.*}} mov x0, #0x1 // =1
# CHECK-NEXT:            {{.*}} ret

# CHECK: <bar.cold.0>:
# CHECK-NEXT: [[ADDR5]]: {{.*}} ret
# CHECK-NEXT: [[ADDR4]]: {{.*}} mov x0, #0x2 // =2
# CHECK-NEXT:            {{.*}} ret
