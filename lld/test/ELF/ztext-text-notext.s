# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld -z notext %t.o -o %t -shared
# RUN: llvm-readobj  -dynamic-table -r %t | FileCheck %s

# CHECK:      Relocations [
# CHECK-NEXT:    Section {{.*}} .rela.dyn {
# CHECK-NEXT:      0x1000 R_X86_64_RELATIVE - 0x1000
# CHECK-NEXT:    }
# CHECK-NEXT:  ]
# CHECK: DynamicSection [
# CHECK: 0x0000000000000016 TEXTREL 0x0

foo:
.quad foo
