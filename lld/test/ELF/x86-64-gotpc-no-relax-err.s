# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: not ld.lld --no-relax -T %t/lds %t/a.o -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

## Test diagnostics for unrelaxed GOTPCRELX overflows. In addition, test that there is no
## `>>> defined in` for linker synthesized __stop_* symbols (there is no
## associated file or linker script line number).

# CHECK:      error: {{.*}}:(.text+0x2): relocation R_X86_64_GOTPCRELX out of range: 2147483666 is not in [-2147483648, 2147483647]; references '__stop_data'
# CHECK-NEXT: >>> defined in <internal>
# CHECK-EMPTY:
# CHECK-NEXT: error: {{.*}}:(.text+0x9): relocation R_X86_64_REX_GOTPCRELX out of range: 2147483659 is not in [-2147483648, 2147483647]; references '__stop_data'
# CHECK-NEXT: >>> defined in <internal>
# CHECK-EMPTY:
# CHECK-NEXT: error: {{.*}}:(.text+0x11): relocation R_X86_64_CODE_4_GOTPCRELX out of range: 2147483651 is not in [-2147483648, 2147483647]; references '__stop_data'
# CHECK-NEXT: >>> defined in <internal>

#--- a.s
  movl __stop_data@GOTPCREL(%rip), %eax  # out of range
  movq __stop_data@GOTPCREL(%rip), %rax  # out of range
  movq __stop_data@GOTPCREL(%rip), %r16  # out of range
  movq __stop_data@GOTPCREL(%rip), %rax  # in range

.section data,"aw",@progbits

#--- lds
SECTIONS {
  .text 0x200000 : { *(.text) }
  .got 0x80200016 : { *(.got) }
}
