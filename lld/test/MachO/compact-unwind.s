# REQUIRES: x86, aarch64
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/my-personality.s -o %t/x86_64-my-personality.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/main.s -o %t/x86_64-main.o
# RUN: %lld -arch x86_64 -pie -lSystem -lc++ %t/x86_64-my-personality.o %t/x86_64-main.o -o %t/x86_64-personality-first
# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --rebase %t/x86_64-personality-first | FileCheck %s --check-prefixes=FIRST,CHECK -D#%x,BASE=0x100000000
# RUN: %lld -arch x86_64 -pie -lSystem -lc++ %t/x86_64-main.o %t/x86_64-my-personality.o -o %t/x86_64-personality-second
# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --rebase %t/x86_64-personality-second | FileCheck %s --check-prefixes=SECOND,CHECK -D#%x,BASE=0x100000000

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin19.0.0 %t/my-personality.s -o %t/arm64-my-personality.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin19.0.0 %t/main.s -o %t/arm64-main.o
# RUN: %lld -arch arm64 -pie -lSystem -lc++ %t/arm64-my-personality.o %t/arm64-main.o -o %t/arm64-personality-first
# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --rebase %t/arm64-personality-first | FileCheck %s --check-prefixes=FIRST,CHECK -D#%x,BASE=0x100000000
# RUN: %lld -arch arm64 -pie -lSystem -lc++ %t/arm64-main.o %t/arm64-my-personality.o -o %t/arm64-personality-second
# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --rebase %t/arm64-personality-second | FileCheck %s --check-prefixes=SECOND,CHECK -D#%x,BASE=0x100000000

# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-watchos %t/my-personality.s -o %t/arm64-32-my-personality.o
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-watchos %t/main.s -o %t/arm64-32-main.o
# RUN: %lld-watchos -pie -lSystem -lc++ %t/arm64-32-my-personality.o %t/arm64-32-main.o -o %t/arm64-32-personality-first
# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --rebase %t/arm64-32-personality-first | FileCheck %s --check-prefixes=FIRST,CHECK -D#%x,BASE=0x4000
# RUN: %lld-watchos -pie -lSystem -lc++ %t/arm64-32-main.o %t/arm64-32-my-personality.o -o %t/arm64-32-personality-second
# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --rebase %t/arm64-32-personality-second | FileCheck %s --check-prefixes=SECOND,CHECK -D#%x,BASE=0x4000

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
# CHECK-DAG:  [[#%x,QUUX:]]       g  F __TEXT,__text _quux
# CHECK-DAG:  [[#%x,FOO:]]        l  F __TEXT,__text _foo
# CHECK-DAG:  [[#%x,BAZ:]]        l  F __TEXT,__text _baz
# CHECK-DAG:  [[#%x,EXCEPTION0:]] g  O __TEXT,__gcc_except_tab _exception0
# CHECK-DAG:  [[#%x,EXCEPTION1:]] g  O __TEXT,__gcc_except_tab _exception1

# CHECK:      Contents of __unwind_info section:
# CHECK:        Personality functions: (count = 2)
# CHECK-DAG:     personality[{{[0-9]+}}]: 0x{{0*}}[[#MY_PERSONALITY-BASE]]
# CHECK-DAG:     personality[{{[0-9]+}}]: 0x{{0*}}[[#GXX_PERSONALITY-BASE]]
# CHECK:        LSDA descriptors:
# CHECK-DAG:     function offset=0x[[#%.8x,FOO-BASE]],  LSDA offset=0x[[#%.8x,EXCEPTION0-BASE]]
# CHECK-DAG:     function offset=0x[[#%.8x,MAIN-BASE]], LSDA offset=0x[[#%.8x,EXCEPTION1-BASE]]
# CHECK:        Second level indices:
# CHECK-DAG:     function offset=0x[[#%.8x,MAIN-BASE]], encoding
# CHECK-DAG:     function offset=0x[[#%.8x,FOO-BASE]], encoding
# CHECK-DAG:     function offset=0x[[#%.8x,BAZ-BASE]], encoding
# CHECK-DAG:     function offset=0x[[#%.8x,QUUX-BASE]], encoding{{.*}}=0x00000000

## Check that we do not add rebase opcodes to the compact unwind section.
# CHECK:      Rebase table:
# CHECK-NEXT: segment      section        address          type
# CHECK-NEXT: __DATA_CONST __got          0x{{[0-9A-F]*}}  pointer
# CHECK-NOT:  __TEXT

#--- my-personality.s
.globl _my_personality, _exception0
.text
.p2align 2
_foo:
  .cfi_startproc
## This will generate a section relocation.
  .cfi_personality 155, _my_personality
  .cfi_lsda 16, _exception0
  .cfi_def_cfa_offset 16
  ret
  .cfi_endproc

.p2align 2
_bar:
  .cfi_startproc
## Check that we dedup references to the same statically-linked personality.
  .cfi_personality 155, _my_personality
  .cfi_lsda 16, _exception0
  .cfi_def_cfa_offset 16
  ret
  .cfi_endproc

.p2align 2
_my_personality:
  ret

.section __TEXT,__gcc_except_tab
_exception0:
  .space 1

#--- main.s
.globl _main, _quux, _my_personality, _exception1

.text
.p2align 2
_main:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_lsda 16, _exception1
  .cfi_def_cfa_offset 16
  ret
  .cfi_endproc

## _quux has no unwind information.
## (In real life, it'd be part of a separate TU that was built with
## -fno-exceptions, while the previous and next TU might be Objective-C++
## which has unwind info for Objective-C).
.p2align 2
_quux:
  ret

.globl _abs
_abs = 4

.p2align 2
_baz:
  .cfi_startproc
## This will generate a symbol relocation. Check that we reuse the personality
## referenced by the section relocation in my_personality.s.
  .cfi_personality 155, _my_personality
  .cfi_lsda 16, _exception1
  .cfi_def_cfa_offset 16
  ret
  .cfi_endproc


.section __TEXT,__gcc_except_tab
_exception1:
  .space 1
