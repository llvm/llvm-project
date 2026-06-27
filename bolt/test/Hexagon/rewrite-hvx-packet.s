## Verify that BOLT can fully rewrite a binary containing HVX (Hexagon Vector
## eXtension) instructions in VLIW packets. HVX vector loads, stores, ALU
## operations, and .new value stores must survive the disassembly-CFG-emit
## round-trip. Real Hexagon binaries make extensive use of HVX, so this is
## a critical code path for BOLT.

# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-linux-musl \
# RUN:   -mcpu=hexagonv68 -mhvx %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe --emit-relocs -e _start
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck --check-prefix=BOLT %s
# RUN: llvm-objdump -d --mattr=+hvx %t.bolt | FileCheck %s

# BOLT-NOT: BOLT-ERROR

  .text
  .globl _start
  .type _start,@function
  .p2align 4
_start:
  call test_hvx_basic
  call test_hvx_newvalue
  call test_hvx_predicated
  call test_hvx_mixed
  jumpr r31
  .size _start, .-_start

##============================================================================
## Basic HVX: vector load, ALU, and store in separate packets.
##============================================================================
# CHECK-LABEL: <test_hvx_basic>:
# CHECK:       vmem(r0+#0x0)
# CHECK:       vmem(r1+#0x0)
# CHECK:       vadd(v0.w,v1.w)
# CHECK:       vmem(r2+#0x0) = v2

  .globl test_hvx_basic
  .type test_hvx_basic,@function
  .p2align 4
test_hvx_basic:
  v0 = vmem(r0 + #0)
  v1 = vmem(r1 + #0)
  v2.w = vadd(v0.w, v1.w)
  vmem(r2 + #0) = v2
  jumpr r31
  .size test_hvx_basic, .-test_hvx_basic

##============================================================================
## HVX .new value store: producer and consumer in the same packet.
## The createBundle code must preserve the producer-consumer ordering
## for the MC encoder's .new distance encoding.
##============================================================================
# CHECK-LABEL: <test_hvx_newvalue>:
# CHECK:       vadd(v0.w,v1.w)
# CHECK:       vmem(r2+#0x0) = v2.new

  .globl test_hvx_newvalue
  .type test_hvx_newvalue,@function
  .p2align 4
test_hvx_newvalue:
  {
    v2.w = vadd(v0.w, v1.w)
    vmem(r2 + #0) = v2.new
  }
  jumpr r31
  .size test_hvx_newvalue, .-test_hvx_newvalue

##============================================================================
## HVX with vector predicates: vector compare producing Q predicate,
## then predicated vector store.
##============================================================================
# CHECK-LABEL: <test_hvx_predicated>:
# CHECK:       vcmp.eq(v0.w,v1.w)
# CHECK:       if (q0) vmem(r2+#0x0) = v0

  .globl test_hvx_predicated
  .type test_hvx_predicated,@function
  .p2align 4
test_hvx_predicated:
  q0 = vcmp.eq(v0.w, v1.w)
  if (q0) vmem(r2 + #0) = v0
  jumpr r31
  .size test_hvx_predicated, .-test_hvx_predicated

##============================================================================
## Mixed scalar and HVX operations in the same function.
## Scalar ALU interleaved with vector load/ALU/store.
##============================================================================
# CHECK-LABEL: <test_hvx_mixed>:
# CHECK:       r0 = add(r0,#0x80)
# CHECK:       vmem(r0+#0x0)
# CHECK:       vadd(v0.w,v0.w)
# CHECK:       vmem(r0+#0x0) = v1

  .globl test_hvx_mixed
  .type test_hvx_mixed,@function
  .p2align 4
test_hvx_mixed:
  r0 = add(r0, #128)
  v0 = vmem(r0 + #0)
  v1.w = vadd(v0.w, v0.w)
  vmem(r0 + #0) = v1
  jumpr r31
  .size test_hvx_mixed, .-test_hvx_mixed
