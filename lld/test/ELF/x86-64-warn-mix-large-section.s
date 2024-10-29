# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/b.s -o %t/b.o
# RUN: ld.lld %t/a.o %t/b.o -o /dev/null 2>&1 | FileCheck %s

# CHECK: warning: incompatible SHF_X86_64_LARGE section flag for 'foo'
# CHECK-NEXT: >>> {{.*}}a.o:(foo): 0x10000003
# CHECK-NEXT: >>> {{.*}}b.o:(foo): 0x3

#--- a.s
.section foo,"awl",@progbits

#--- b.s
.section foo,"aw",@progbits

