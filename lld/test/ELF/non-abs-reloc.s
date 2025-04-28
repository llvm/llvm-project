// REQUIRES: x86
// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: llvm-mc -filetype=obj -triple=i386 --defsym X86_32=1 asm -o a.o
// RUN: ld.lld -T lds a.o -o a 2>&1 | FileCheck %s --implicit-check-not=warning:
// CHECK:      warning: {{.*}}.o:(.nonalloc1+0x1): has non-ABS relocation R_386_PC32 against symbol '_start'
// CHECK-NEXT: warning: {{.*}}.o:(.nonalloc1+0x6): has non-ABS relocation R_386_PC32 against symbol 'ifunc'
// CHECK-NEXT: warning: {{.*}}.o:(.nonalloc1+0xa): has non-ABS relocation R_386_PC32 against symbol ''

// RUN: llvm-objdump -D --no-show-raw-insn a | FileCheck --check-prefix=DISASM %s
// DISASM:      Disassembly of section .nonalloc:
// DISASM-EMPTY:
// DISASM-NEXT: <.nonalloc>:
// DISASM-NEXT:   0: nop
// DISASM-NEXT:   1: call{{.}} 0x0
// DISASM-NEXT:   6: call{{.}} 0x5

/// There is currently no error for -r. See also https://github.com/ClangBuiltLinux/linux/issues/1937
// RUN: ld.lld -T lds -r a.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=REL-R --implicit-check-not=warning:
// REL-R: warning: {{.*}}:(.nonalloc1+0xa): has non-ABS relocation R_386_PC32 against symbol ''

// RUN: llvm-mc -filetype=obj -triple=x86_64 asm -o b.o
// RUN: ld.lld -T lds b.o -o b 2>&1 | FileCheck %s --check-prefix=CHECK2 --implicit-check-not=warning:
// RUN: llvm-objdump -D --no-show-raw-insn b | FileCheck --check-prefix=DISASM %s
// RUN: ld.lld -T lds -r b.o -o /dev/null --fatal-warnings
// CHECK2:      warning: {{.*}}.o:(.nonalloc1+0x1): has non-ABS relocation R_X86_64_PC32 against symbol '_start'
// CHECK2-NEXT: warning: {{.*}}.o:(.nonalloc1+0x6): has non-ABS relocation R_X86_64_PC32 against symbol 'ifunc'
// CHECK2-NEXT: warning: {{.*}}.o:(.nonalloc1+0xa): has non-ABS relocation R_X86_64_PC32 against symbol ''

//--- lds
SECTIONS {
  .nonalloc 0 : { *(.nonalloc*) }
}
//--- asm
.globl _start
_start:
.L0:
  nop

resolver: ret
.type ifunc, @gnu_indirect_function
.set ifunc, resolver

.section .nonalloc0
  nop

.section .nonalloc1
  .byte 0xe8
  .long _start - . - 4
  .byte 0xe8
  .long ifunc - .
  .long .nonalloc0 - .

// GCC may relocate DW_AT_GNU_call_site_value with R_386_GOTOFF.
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98946
.ifdef X86_32
.section .debug_random
  .long .L0@gotoff
.endif
