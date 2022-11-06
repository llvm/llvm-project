// REQUIRES: aarch64
// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=aarch64-pc-linux
// RUN: llvm-mc %p/Inputs/aarch64-tls-gdie.s -o %t2.o -filetype=obj -triple=aarch64-pc-linux
// RUN: ld.lld %t2.o -o %t2.so -shared -soname=t2.so
// RUN: ld.lld --hash-style=sysv %t.o %t2.so -o %t
// RUN: llvm-readobj -S %t | FileCheck --check-prefix=SEC %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck %s

        .globl  _start
_start:
        nop
        adrp    x0, :tlsdesc:a
        ldr     x1, [x0, :tlsdesc_lo12:a]
        add     x0, x0, :tlsdesc_lo12:a
        .tlsdesccall a
        blr     x1

// SEC:      Name: .got
// SEC-NEXT: Type: SHT_PROGBITS
// SEC-NEXT: Flags [
// SEC-NEXT:   SHF_ALLOC
// SEC-NEXT:   SHF_WRITE
// SEC-NEXT: ]
// SEC-NEXT: Address: 0x220300

// page(0x220300) - page(0x21023c) = 65536
// 0x23c = 768

// CHECK:      <_start>:
// CHECK-NEXT: 210238: nop
// CHECK-NEXT: 21023c: adrp    x0, 0x220000
// CHECK-NEXT: 210240: ldr     x0, [x0, #768]
// CHECK-NEXT: 210244: nop
// CHECK-NEXT: 210248: nop
