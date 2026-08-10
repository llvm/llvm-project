# REQUIRES: x86
## Test --emit-relocs handles .debug*

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --emit-relocs %t.o -o %t
# RUN: llvm-readobj -r %t | FileCheck %s
# RUN: ld.lld --emit-relocs --strip-debug %t.o -o %t.no
# RUN: llvm-readobj -r %t.no | FileCheck --check-prefix=NO %s

# CHECK:      Section {{.*}} .rela.debug_info {
# CHECK-NEXT:   R_X86_64_64 .text 0x0
# CHECK-NEXT: }

# NO:      Relocations [
# NO-NEXT: ]

foo:

.section .debug_info
.quad foo
