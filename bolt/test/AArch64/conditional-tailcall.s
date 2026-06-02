# Bolt can handle conditional tail calls.

# RUN: %clang %cflags -Wl,-q %s -o %t
# RUN: llvm-bolt %t -o %t.bolt --skip-funcs foo
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

  .text
  .globl foo
  .type foo, %function
foo:

test_cbz:
  cbz xzr, bar
test_cbnz:
  cbnz xzr, bar
test_tbz:
  tbz xzr, #0, bar
test_tbnz:
  tbnz xzr, #0, bar
test_beq:
  cmp wzr, wzr
  b.eq bar
test_bne:
  cmp wzr, wzr
  b.ne bar

  .globl bar
  .type bar, %function
bar:
  ret  xzr

# CHECK: Disassembly of section .bolt.org.text:
#
# CHECK:      <test_cbz>:
# CHECK-NEXT:            {{.*}} cbz xzr, 0x[[ADDR:[0-9a-f]+]] <bar>
# CHECK:      <test_cbnz>:
# CHECK-NEXT:            {{.*}} cbnz xzr, 0x[[ADDR]] <bar>
# CHECK:      <test_tbz>:
# CHECK-NEXT:            {{.*}} tbz wzr, #0x0, 0x[[ADDR]] <bar>
# CHECK:      <test_tbnz>:
# CHECK-NEXT:            {{.*}} tbnz wzr, #0x0, 0x[[ADDR]] <bar>
# CHECK:      <test_beq>:
# CHECK-NEXT:            {{.*}} cmp wzr, wzr
# CHECK-NEXT:            {{.*}} b.eq 0x[[ADDR]] <bar>
# CHECK:      <test_bne>:
# CHECK-NEXT:            {{.*}} cmp wzr, wzr
# CHECK-NEXT:            {{.*}} b.ne 0x[[ADDR]] <bar>
# CHECK:      <bar>:
# CHECK-NEXT: [[ADDR]]: {{.*}} ret xzr
