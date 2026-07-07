## Verify that BOLT can fully rewrite a binary containing new-value store
## operations where the store operand is produced and consumed within the
## same VLIW packet. BOLT must preserve producer-consumer ordering within
## the BUNDLE MCInst for the MC code emitter's .new distance encoding.

# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-linux-musl %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe --emit-relocs -e _start
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck --check-prefix=BOLT %s
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

# BOLT-NOT: BOLT-ERROR

# CHECK-LABEL: <test_nvstore>:
# CHECK:       r0 = add(r1,r2)
# CHECK-NEXT:  memw(r3+#0x0) = r0.new

  .text
  .globl _start
  .type _start,@function
  .p2align 4
_start:
    call test_nvstore
    jumpr r31
  .size _start, .-_start

  .globl test_nvstore
  .type test_nvstore,@function
  .p2align 4
test_nvstore:
  {
    r0 = add(r1, r2)
    memw(r3+#0) = r0.new
  }
    jumpr r31
  .size test_nvstore, .-test_nvstore
