# REQUIRES: aarch64, x86
## Test we resolve symbolic relocations in .debug_* sections to a tombstone
## value if the referenced symbol is discarded (--gc-sections, non-prevailing
## section group, SHF_EXCLUDE, /DISCARD/, etc).

# RUN: echo '.globl _start; _start: call group' | llvm-mc -filetype=obj -triple=x86_64 - -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t1.o
# RUN: ld.lld --emit-relocs --gc-sections %t.o %t1.o %t1.o -o %t
# RUN: llvm-objdump -s %t | FileCheck %s
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=REL

# RUN: echo '.globl _start; _start: bl group' | llvm-mc -filetype=obj -triple=aarch64 - -o %t.a64.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t1.a64.o
# RUN: ld.lld --emit-relocs --gc-sections %t.a64.o %t1.a64.o %t1.a64.o -o %t.a64
# RUN: llvm-objdump -s %t.a64 | FileCheck %s

# CHECK:      Contents of section .debug_loc:
# CHECK-NEXT:  0000 01000000 00000000 01000000 00000000
# CHECK:      Contents of section .debug_ranges:
# CHECK-NEXT:  0000 01000000 00000000 01000000 00000000
# CHECK:      Contents of section .debug_addr:
# CHECK-NEXT:  0000 {{.*}}00 00000000 {{.*}}00 00000000
# CHECK-NEXT:  0010 00000000 00000000 {{.*}}00 00000000
# CHECK:      Contents of section .debug_names:
# CHECK-NEXT:  0000 00000000 00000000 00000000 ffffffff .
# CHECK-NEXT:  0010 ffffffff ffffffff                   .
# CHECK:      Contents of section .debug_foo:
# CHECK-NEXT:  0000 00000000 00000000 00000000 00000000
# CHECK-NEXT:  0010 00000000 00000000 00000000 00000000

# REL:      Relocations [
# REL-NEXT:   .rela.text {
# REL-NEXT:     0x201121 R_X86_64_PLT32 group 0xFFFFFFFFFFFFFFFC
# REL-NEXT:   }
# REL-NEXT:   .rela.debug_loc {
# REL-NEXT:     0x0 R_X86_64_NONE - 0x8
# REL-NEXT:     0x8 R_X86_64_NONE - 0x8
# REL-NEXT:   }
# REL-NEXT:   .rela.debug_ranges {
# REL-NEXT:     0x0 R_X86_64_NONE - 0x10
# REL-NEXT:     0x8 R_X86_64_NONE - 0x10
# REL-NEXT:   }
# REL-NEXT:   .rela.debug_addr {
# REL-NEXT:     0x0 R_X86_64_64 .text 0x1D
# REL-NEXT:     0x8 R_X86_64_64 group 0x20
# REL-NEXT:     0x10 R_X86_64_NONE - 0x18
# REL-NEXT:     0x18 R_X86_64_64 group 0x20
# REL-NEXT:   }
# REL-NEXT:   .rela.debug_names {
# REL-NEXT:     0x0 R_X86_64_32 .debug_info 0x0
# REL-NEXT:     0x4 R_X86_64_64 .debug_info 0x0
# REL-NEXT:     0xC R_X86_64_NONE - 0x0
# REL-NEXT:     0x10 R_X86_64_NONE - 0x0
# REL-NEXT:   }
# REL-NEXT:   .rela.debug_foo {
# REL-NEXT:     0x0 R_X86_64_NONE - 0x8
# REL-NEXT:     0x8 R_X86_64_NONE - 0x8
# REL-NEXT:     0x10 R_X86_64_NONE - 0x8
# REL-NEXT:     0x18 R_X86_64_NONE - 0x8
# REL-NEXT:   }
# REL-NEXT: ]

## -z dead-reloc-in-nonalloc= can override the tombstone value.
# RUN: ld.lld --gc-sections -z dead-reloc-in-nonalloc=.debug_loc=42 %t.o %t1.o %t1.o -o %t42
# RUN: llvm-objdump -s %t42 | FileCheck %s --check-prefix=OVERRIDE

# OVERRIDE:      Contents of section .debug_loc:
# OVERRIDE-NEXT:  0000 2a000000 00000000 2a000000 00000000

.section .text.1,"ax"
  .byte 0
.section .text.2,"axe"
  .byte 0
.section .text.3,"axG",@progbits,group,comdat
.globl group
group:
  .byte 0

## Resolved to UINT64_C(1), with the addend ignored.
## UINT64_C(-1) is a reserved value (base address selection entry) which can't be used.
.section .debug_loc
  .quad .text.1+8
.section .debug_ranges
  .quad .text.2+16

.section .debug_addr
## .text.3 is a local symbol. The symbol defined in a non-prevailing group is
## discarded. Resolved to UINT64_C(0).
  .quad .text.3+24
## group is a non-local symbol. The relocation from the second %t1.o gets
## resolved to the prevailing copy.
  .quad group+32

.section  .debug_info,"G",@progbits,5657452045627120676,comdat
.Ltu_begin0:

.section .debug_names
## .debug_names may reference a local type unit defined in a COMDAT .debug_info
## section (-g -gpubnames -fdebug-types-section). If the referenced section is
## non-prevailing, resolve to UINT32_MAX.
.long .Ltu_begin0
## ... or UINT64_MAX for DWARF64.
.quad .Ltu_begin0

.section .debug_foo
  .quad .text.1+8

## We only deal with DW_FORM_addr. Don't special case short-range absolute
## relocations. Treat them like regular absolute relocations referencing
## discarded symbols, which are resolved to the addend.
  .long .text.1+8
  .long 0
