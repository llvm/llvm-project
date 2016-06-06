// REQUIRES: aarch64
// RUN: llvm-mc %s -o %t.o -triple aarch64-pc-linux -filetype=obj
// RUN: ld.lld %t.o -o %t.so -shared
// RUN: llvm-readobj -s %t.so | FileCheck --check-prefix=SEC %s
// RUN: llvm-objdump -d %t.so | FileCheck %s

foo:
        adrp    x0, :tlsdesc:bar
        ldr     x1, [x0, :tlsdesc_lo12:bar]
        add     x0, x0, :tlsdesc_lo12:bar
        .tlsdesccall bar
        blr     x1


        .section        .tdata,"awT",@progbits
bar:
        .word   42


// SEC:      Name: .got
// SEC-NEXT: Type: SHT_PROGBITS
// SEC-NEXT: Flags [
// SEC-NEXT:   SHF_ALLOC
// SEC-NEXT:   SHF_WRITE
// SEC-NEXT: ]
// SEC-NEXT: Address: 0x2098
// SEC-NEXT: Offset: 0x2098
// SEC-NEXT: Size: 16

// page(0x2098) - page(0x1000) = 4096
// 0x98 = 152

// CHECK:      foo:
// CHECK-NEXT: 1000: {{.*}} adrp x0, #4096
// CHECK-NEXT: 1004: {{.*}} ldr  x1, [x0, #152]
// CHECK-NEXT: 1008: {{.*}} add  x0, x0, #152
// CHECK-NEXT: 100c: {{.*}} blr  x1
