# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: not ld.lld -T %t/lds %t/a.o -o /dev/null 2>&1 | FileCheck %s

## Test diagnostics for GOTPCRELX overflows. In addition, test that there is no
## `>>> defined in` for linker synthesized __stop_* symbols (there is no
## associated file or linker script line number).

# CHECK:      error: {{.*}}:(.text+0x2): relocation R_X86_64_GOTPCRELX out of range: 2147483655 is not in [-2147483648, 2147483647]; references '__stop_data'
# CHECK-NEXT: error: {{.*}}:(.text+0x9): relocation R_X86_64_REX_GOTPCRELX out of range: 2147483648 is not in [-2147483648, 2147483647]; references '__stop_data'
# CHECK-NOT: error:

#--- a.s
  movl __stop_data@GOTPCREL(%rip), %eax  # out of range
  movq __stop_data@GOTPCREL(%rip), %rax  # out of range
  movq __stop_data@GOTPCREL(%rip), %rax  # in range

.section data,"aw",@progbits
.space 13

#--- lds
SECTIONS {
  .text 0x200000 : { *(.text) }
  data 0x80200000 : { *(data) }
}
