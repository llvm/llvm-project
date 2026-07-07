## Verify that BOLT can fully rewrite new-value compare-and-jump (NCJ)
## instructions where a register value is loaded and immediately used in
## a compare-and-jump within the same packet. The MC encoder must find
## the producer instruction's position within the bundle for the .new
## operand distance encoding.

# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-linux-musl %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe --emit-relocs -e _start
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck --check-prefix=BOLT %s
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

# BOLT-NOT: BOLT-ERROR

# CHECK-LABEL: <test_nvjump>:
# CHECK:       r0 = memw(r1+#0x0)
# CHECK:       if (cmp.eq(r0.new,#0x0)) jump:nt

  .text
  .globl _start
  .type _start,@function
  .p2align 4
_start:
    call test_nvjump
    jumpr r31
  .size _start, .-_start

  .globl test_nvjump
  .type test_nvjump,@function
  .p2align 4
test_nvjump:
  {
    r0 = memw(r1+#0)
    if (cmp.eq(r0.new,#0)) jump:nt .Ltarget
  }
    r0 = #1
    jumpr r31
.Ltarget:
    r0 = #0
    jumpr r31
  .size test_nvjump, .-test_nvjump
