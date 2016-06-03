// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-linux %s -o %t.o
// RUN: ld.lld -shared %t.o -o %t.so
// RUN: llvm-objdump -d %t.so | FileCheck %s
// RUN: llvm-readobj -r %t.so | FileCheck --check-prefix=REL %s

        adrp    x0, :tlsdesc:a
        ldr     x1, [x0, :tlsdesc_lo12:a]
        add     x0, x0, :tlsdesc_lo12:a
        .tlsdesccall a
        blr     x1

// CHECK:      1000: {{.*}}  adrp    x0, #4096
// CHECK-NEXT: 1004: {{.*}}  ldr     x1, [x0, #144]
// CHECK-NEXT: 1008: {{.*}}  add     x0, x0, #144
// CHECK-NEXT: 100c: {{.*}}  blr     x1

// 0x1000 + 4096 + 144 = 0x2090

// REL:      Relocations [
// REL-NEXT:   Section (4) .rela.dyn {
// REL-NEXT:     0x2090 R_AARCH64_TLSDESC a 0x0
// REL-NEXT:   }
// REL-NEXT: ]
