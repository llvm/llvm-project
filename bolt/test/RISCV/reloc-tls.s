// RUN: llvm-mc -triple riscv64 -filetype obj -o %t.o %s
// RUN: ld.lld --emit-relocs -o %t %t.o
// RUN: llvm-bolt --print-cfg --print-only=tls_le,tls_ie -o %t.null %t \
// RUN:    | FileCheck %s

// CHECK-LABEL: Binary Function "tls_le{{.*}}" after building cfg {
// CHECK:      lui a5, 0x0
// CHECK-NEXT: add a5, a5, tp
// CHECK-NEXT: lw t0, 0x0(a5)
// CHECK-NEXT: sw t0, 0x0(a5)

// CHECK-LABEL: Binary Function "tls_ie" after building cfg {
// CHECK-LABEL: .LBB01
// CHECK:      auipc a0, %pcrel_hi(__BOLT_got_zero+{{[0-9]+}})
// CHECK-NEXT: ld a0, %pcrel_lo(.Ltmp0)(a0)
    .text
    .globl tls_le, _start
    .p2align 2
tls_le:
_start:
    nop
    lui a5, %tprel_hi(i)
    add a5, a5, tp, %tprel_add(i)
    lw t0, %tprel_lo(i)(a5)
    sw t0, %tprel_lo(i)(a5)
    ret
    .size _start, .-_start

    .globl tls_ie
    .p2align 2
tls_ie:
    nop
    la.tls.ie a0, i
    ret
    .size tls_ie, .-tls_ie

    .section .tbss,"awT",@nobits
    .type i,@object
    .globl i
    .p2align 3
i:
    .quad 0
    .size i, .-i

