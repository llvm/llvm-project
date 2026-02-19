// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
// RUN: not ld.lld %t -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:
// CHECK: error: {{.*}}:(.text+0x4): relocation R_X86_64_TPOFF32 out of range: {{.*}}; references 'a'

.global _start
_start:
        movl %fs:a@tpoff, %eax
.global a
.section        .tbss,"awT",@nobits
a:
.zero 0x80000001
