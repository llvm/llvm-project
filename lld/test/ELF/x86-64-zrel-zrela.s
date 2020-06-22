# REQUIRES: x86
## The x86-64 psABI uses Elf64_Rela relocation entries. We produce
## Elf64_Rel dynamic relocations by default, but can use Elf64_Rel with -z rel.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readobj -d -r %t.so | FileCheck --check-prefix=RELA %s
# RUN: ld.lld -shared -z rela %t.o -o %t1.so
# RUN: llvm-readobj -d -r %t1.so | FileCheck --check-prefix=RELA %s

# RELA:      RELA      {{.*}}
# RELA-NEXT: RELASZ    72 (bytes)
# RELA-NEXT: RELAENT   24 (bytes)
# RELA-NEXT: RELACOUNT 1
# RELA-NEXT: JMPREL    {{.*}}
# RELA-NEXT: PLTRELSZ  24 (bytes)
# RELA-NEXT: PLTGOT    {{.*}}
# RELA-NEXT: PLTREL    RELA
# RELA:      .rela.dyn {
# RELA-NEXT:   R_X86_64_RELATIVE - 0x3428
# RELA-NEXT:   R_X86_64_GLOB_DAT func 0x0
# RELA-NEXT:   R_X86_64_64 _start 0x2A
# RELA-NEXT: }
# RELA-NEXT: .rela.plt {
# RELA-NEXT:   R_X86_64_JUMP_SLOT func 0x0
# RELA-NEXT: }

# RUN: ld.lld -shared -z rela -z rel %t.o -o %t2.so
# RUN: llvm-readobj -d -r -x .data %t2.so | FileCheck --check-prefix=REL %s

# REL:      REL      {{.*}}
# REL-NEXT: RELSZ    48 (bytes)
# REL-NEXT: RELENT   16 (bytes)
# REL-NEXT: RELCOUNT 1
# REL-NEXT: JMPREL   {{.*}}
# REL-NEXT: PLTRELSZ 16 (bytes)
# REL-NEXT: PLTGOT   {{.*}}
# REL-NEXT: PLTREL   REL{{$}}
# REL:      .rel.dyn {
# REL-NEXT:   R_X86_64_RELATIVE - 0x0
# REL-NEXT:   R_X86_64_GLOB_DAT func 0x0
# REL-NEXT:   R_X86_64_64 _start 0
# REL-NEXT: }
# REL-NEXT: .rel.plt {
# REL-NEXT:   R_X86_64_JUMP_SLOT func 0x0
# REL-NEXT: }

# REL:      Hex dump of section '.data':
# REL-NEXT: 0x00003408 08340000 00000000 2a000000 00000000

.globl _start
_start:
  call func@PLT
  movq func@GOTPCREL(%rip), %rax

.data
  .quad .data
  .quad _start+42
