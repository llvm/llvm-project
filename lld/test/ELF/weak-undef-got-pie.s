# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/dummy-shared.s -o %t1.o
# RUN: ld.lld %t1.o -shared -o %t1.so
# RUN: llvm-mc -filetype=obj -x86-relax-relocations=false -triple=x86_64 %s -o %t.o

# RUN: ld.lld -pie %t.o %t1.so -o %t
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=RELOCS %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=DISASM %s

# RELOCS:      Relocations [
# RELOCS-NEXT:   Section ({{.*}}) .rela.dyn {
# RELOCS-NEXT:     R_X86_64_GLOB_DAT foo 0x0
# RELOCS-NEXT:   }
# RELOCS-NEXT: ]

.weak foo

.globl _start
_start:
# DISASM: <_start>:
# DISASM-NEXT: movq {{.*}}(%rip), %rax
mov foo@gotpcrel(%rip), %rax
