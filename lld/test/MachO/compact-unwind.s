# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %s -o %t.o
# RUN: %lld -pie -lSystem -lc++ %t.o -o %t
# RUN: llvm-objdump --macho --unwind-info --indirect-symbols --rebase %t | FileCheck %s

# CHECK:      Indirect symbols for (__DATA_CONST,__got)
# CHECK-NEXT: address                    index name
# CHECK-DAG:  0x[[#%x,GXX_PERSONALITY:]] [[#]] ___gxx_personality_v0
# CHECK-DAG:  0x[[#%x,MY_PERSONALITY:]]  LOCAL

# CHECK:      Contents of __unwind_info section:
# CHECK:        Personality functions: (count = 2)
# CHECK-NEXT:     personality[1]: 0x{{0*}}[[#MY_PERSONALITY-0x100000000]]
# CHECK-NEXT:     personality[2]: 0x{{0*}}[[#GXX_PERSONALITY-0x100000000]]

## Check that we do not add rebase opcodes to the compact unwind section.
# CHECK:      Rebase table:
# CHECK-NEXT: segment      section        address          type
# CHECK-NEXT: __DATA_CONST __got          0x{{[0-9a-f]*}}  pointer
# CHECK-NEXT: __DATA_CONST __got          0x{{[0-9a-f]*}}  pointer
# CHECK-EMPTY:

.globl _main, _foo, _my_personality, _bar

.text
_foo:
  .cfi_startproc
  .cfi_personality 155, _my_personality
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_bar:
  .cfi_startproc
## Check that we dedup references to the same statically-linked personality.
  .cfi_personality 155, _my_personality
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_main:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_my_personality:
  retq
