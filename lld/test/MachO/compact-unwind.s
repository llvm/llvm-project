# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %s -o %t.o
# RUN: %lld -pie -lSystem -lc++ %t.o -o %t
# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --rebase %t | FileCheck %s

# CHECK:      Indirect symbols for (__DATA_CONST,__got)
# CHECK-NEXT: address                    index name
# CHECK-DAG:  0x[[#%x,GXX_PERSONALITY:]] [[#]] ___gxx_personality_v0
# CHECK-DAG:  0x[[#%x,MY_PERSONALITY:]]  LOCAL

# CHECK:      SYMBOL TABLE:
# CHECK-DAG:  [[#%x,MAIN:]]       g  F __TEXT,__text _main
# CHECK-DAG:  [[#%x,FOO:]]        g  F __TEXT,__text _foo
# CHECK-DAG:  [[#%x,EXCEPTION0:]] g  O __TEXT,__gcc_except_tab _exception0
# CHECK-DAG:  [[#%x,EXCEPTION1:]] g  O __TEXT,__gcc_except_tab _exception1

# CHECK:      Contents of __unwind_info section:
# CHECK:        Personality functions: (count = 2)
# CHECK-NEXT:     personality[1]: 0x{{0*}}[[#MY_PERSONALITY-0x100000000]]
# CHECK-NEXT:     personality[2]: 0x{{0*}}[[#GXX_PERSONALITY-0x100000000]]
# CHECK:        LSDA descriptors:
# CHECK-NEXT:     [0]: function offset=0x{{0*}}[[#FOO-0x100000000]],  LSDA offset=0x{{0*}}[[#EXCEPTION0-0x100000000]]
# CHECK-NEXT:     [1]: function offset=0x{{0*}}[[#MAIN-0x100000000]], LSDA offset=0x{{0*}}[[#EXCEPTION1-0x100000000]]

## Check that we do not add rebase opcodes to the compact unwind section.
# CHECK:      Rebase table:
# CHECK-NEXT: segment      section        address          type
# CHECK-NEXT: __DATA_CONST __got          0x{{[0-9a-f]*}}  pointer
# CHECK-NEXT: __DATA_CONST __got          0x{{[0-9a-f]*}}  pointer
# CHECK-EMPTY:

.globl _main, _foo, _my_personality, _bar, _exception0, _exception1

.text
_foo:
  .cfi_startproc
  .cfi_personality 155, _my_personality
  .cfi_lsda 16, _exception0
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_bar:
  .cfi_startproc
## Check that we dedup references to the same statically-linked personality.
  .cfi_personality 155, _my_personality
  .cfi_lsda 16, _exception0
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_main:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_lsda 16, _exception1
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_my_personality:
  retq

.section __TEXT,__gcc_except_tab
_exception0:
  .space 1
_exception1:
  .space 1
