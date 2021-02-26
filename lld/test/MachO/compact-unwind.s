# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/my_personality.s -o %t/my_personality.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/main.s -o %t/main.o
# RUN: %lld -pie -lSystem -lc++ %t/my_personality.o %t/main.o -o %t/personality-first
# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --rebase %t/personality-first | FileCheck %s --check-prefixes=FIRST,CHECK
# RUN: %lld -pie -lSystem -lc++ %t/main.o %t/my_personality.o -o %t/personality-second
# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --rebase %t/personality-second | FileCheck %s --check-prefixes=SECOND,CHECK

# FIRST:      Indirect symbols for (__DATA_CONST,__got)
# FIRST-NEXT: address                    index name
# FIRST-DAG:  0x[[#%x,GXX_PERSONALITY:]] [[#]] ___gxx_personality_v0
# FIRST-DAG:  0x[[#%x,MY_PERSONALITY:]]  LOCAL

# SECOND:      Indirect symbols for (__DATA_CONST,__got)
# SECOND-NEXT: address                    index name
# SECOND-DAG:  0x[[#%x,GXX_PERSONALITY:]] [[#]] ___gxx_personality_v0
# SECOND-DAG:  0x[[#%x,MY_PERSONALITY:]]  [[#]] _my_personality

# CHECK:      SYMBOL TABLE:
# CHECK-DAG:  [[#%x,MAIN:]]       g  F __TEXT,__text _main
# CHECK-DAG:  [[#%x,FOO:]]        l  F __TEXT,__text _foo
# CHECK-DAG:  [[#%x,EXCEPTION0:]] g  O __TEXT,__gcc_except_tab _exception0
# CHECK-DAG:  [[#%x,EXCEPTION1:]] g  O __TEXT,__gcc_except_tab _exception1

# CHECK:      Contents of __unwind_info section:
# CHECK:        Personality functions: (count = 2)
# CHECK-DAG:     personality[{{[0-9]+}}]: 0x{{0*}}[[#MY_PERSONALITY-0x100000000]]
# CHECK-DAG:     personality[{{[0-9]+}}]: 0x{{0*}}[[#GXX_PERSONALITY-0x100000000]]
# CHECK:        LSDA descriptors:
# CHECK-DAG:     function offset=0x{{0*}}[[#FOO-0x100000000]],  LSDA offset=0x{{0*}}[[#EXCEPTION0-0x100000000]]
# CHECK-DAG:     function offset=0x{{0*}}[[#MAIN-0x100000000]], LSDA offset=0x{{0*}}[[#EXCEPTION1-0x100000000]]

## Check that we do not add rebase opcodes to the compact unwind section.
# CHECK:      Rebase table:
# CHECK-NEXT: segment      section        address          type
# CHECK-NEXT: __DATA_CONST __got          0x{{[0-9a-f]*}}  pointer
# CHECK-NOT:  __TEXT

#--- my_personality.s
.globl _my_personality, _exception0
.text
_foo:
  .cfi_startproc
## This will generate a section relocation.
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

_my_personality:
  retq

.section __TEXT,__gcc_except_tab
_exception0:
  .space 1

#--- main.s
.globl _main, _my_personality, _exception1

.text
_main:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_lsda 16, _exception1
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_baz:
  .cfi_startproc
## This will generate a symbol relocation. Check that we reuse the personality
## referenced by the section relocation in my_personality.s.
  .cfi_personality 155, _my_personality
  .cfi_lsda 16, _exception1
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc


.section __TEXT,__gcc_except_tab
_exception1:
  .space 1
