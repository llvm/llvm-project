# REQUIRES: x86
## Test HIP specific sections layout.

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux --defsym=HIP_SECTIONS=1 --defsym=NON_HIP_SECTIONS=1 %s -o %t.o
# RUN: ld.lld %t.o -o %t.out
# RUN: llvm-readobj --sections %t.out | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux --defsym=NON_HIP_SECTIONS=1 %s -o %t.1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux --defsym=HIP_SECTIONS=1 %s -o %t.2.o
# RUN: ld.lld %t.1.o %t.2.o -o %t.1.s.out
# RUN: llvm-readobj --sections %t.1.s.out | FileCheck %s
# RUN: ld.lld %t.2.o %t.1.o -o %t.2.s.out
# RUN: llvm-readobj --sections %t.2.s.out | FileCheck %s

.ifdef HIP_SECTIONS
.section .hipFatBinSegment,"aw",@progbits; .space 1
.section .hip_gpubin_handle,"aw",@progbits; .space 1
.section .hip_fatbin,"a",@progbits; .space 1
.endif

.ifdef NON_HIP_SECTIONS
.global _start
.text
_start:
.section .bss,"aw",@nobits; .space 1
.section .debug_info,"",@progbits
.section .debug_line,"",@progbits
.section .debug_str,"MS",@progbits,1
.endif

# Check that the HIP sections are placed towards the end but before non allocated sections

// CHECK: Name: .text
// CHECK: Name: .bss
// CHECK: Name: .hipFatBinSegment
// CHECK: Name: .hip_gpubin_handle
// CHECK: Name: .hip_fatbin
// CHECK: Name: .debug_info
// CHECK: Name: .debug_line
// CHECK: Name: .debug_str

