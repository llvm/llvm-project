# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -lSystem -o %t.out %t.o -dead_strip
# RUN: llvm-otool -l %t.out | FileCheck --check-prefix=SECT %s
# RUN: llvm-otool -vs __TEXT __cstring %t.out | FileCheck %s

# SECT:      sectname __cstring
# SECT-NEXT: segname __TEXT
# SECT-NEXT: addr
# SECT-NEXT: size
# SECT-NEXT: offset
# SECT-NEXT: align 2^4 (16)

# CHECK: 0 \303Q043\005\376\334\272\230vT2\020\001
# CHECK: 8 def

.section __TEXT,__cstring,cstring_literals
.globl _foo
_foo:  # Dead. External, has symbol table entry, gets stripped.
  .asciz "asdf"

.globl _hi
_hi:
  .asciz "hi" # External, has symbol table entry.

.p2align 4
L_internal_aligned_16: # Has no symbol table entry.
  .asciz "\303Q043\005\376\334\272\230vT2\020\001"

L_internal_nonaligned:
  .asciz "abc"

.p2align 3
L_internal_aligned_8:
  .asciz "def"

.text
.globl _main
_main:
  movq _hi(%rip), %rax
  movq L_internal_nonaligned(%rip), %rax
  movq L_internal_aligned_8(%rip), %rax
  movaps L_internal_aligned_16(%rip), %xmm0
  retq

.subsections_via_symbols
