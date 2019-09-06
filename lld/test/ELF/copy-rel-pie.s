# REQUIRES: x86

## Test copy relocations can be created for -pie.

# RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
# RUN: llvm-mc %p/Inputs/copy-rel-pie.s -o %t2.o -filetype=obj -triple=x86_64-pc-linux
# RUN: ld.lld %t2.o -o %t2.so -shared
# RUN: ld.lld %t.o %t2.so -o %t -pie
# RUN: llvm-readobj -r %t | FileCheck %s
# RUN: llvm-objdump -d %t | FileCheck --check-prefix=DISASM %s

.global _start
_start:
        .byte 0xe8
        .long bar - . -4
        .byte 0xe8
        .long foo - . -4

// CHECK:      Relocations [
// CHECK-NEXT:   .rela.dyn {
// CHECK-NEXT:     R_X86_64_COPY foo 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   .rela.plt {
// CHECK-NEXT:     R_X86_64_JUMP_SLOT bar 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// DISASM:      _start:
// DISASM-NEXT:   callq   {{.*}} <bar@plt>
// DISASM-NEXT:   callq   {{.*}} <foo>
