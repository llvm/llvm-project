# REQUIRES: x86, aarch64
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 -emit-compact-unwind-non-canonical=true %t/my-personality.s -o %t/x86_64-my-personality.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 -emit-compact-unwind-non-canonical=true %t/main.s -o %t/x86_64-main.o
# RUN: %lld -arch x86_64 -lSystem -lc++ %t/x86_64-my-personality.o %t/x86_64-main.o -o %t/x86_64-personality-first
# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --rebase %t/x86_64-personality-first | FileCheck %s --check-prefixes=FIRST,CHECK -D#%x,BASE=0x100000000 -DSEG=__TEXT
# RUN: %lld -dead_strip -arch x86_64 -lSystem -lc++ %t/x86_64-main.o %t/x86_64-my-personality.o -o %t/x86_64-personality-second
# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --rebase %t/x86_64-personality-second | FileCheck %s --check-prefixes=SECOND,CHECK -D#%x,BASE=0x100000000 -DSEG=__TEXT

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin19.0.0 -emit-compact-unwind-non-canonical=true %t/my-personality.s -o %t/arm64-my-personality.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin19.0.0 -emit-compact-unwind-non-canonical=true %t/main.s -o %t/arm64-main.o
# RUN: %lld -arch arm64 -lSystem -lc++ %t/arm64-my-personality.o %t/arm64-main.o -o %t/arm64-personality-first
# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --rebase %t/arm64-personality-first | FileCheck %s --check-prefixes=FIRST,CHECK -D#%x,BASE=0x100000000 -DSEG=__TEXT
# RUN: %lld -dead_strip -arch arm64 -lSystem -lc++ %t/arm64-main.o %t/arm64-my-personality.o -o %t/arm64-personality-second
# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --rebase %t/arm64-personality-second | FileCheck %s --check-prefixes=SECOND,CHECK -D#%x,BASE=0x100000000 -DSEG=__TEXT

# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-watchos -emit-compact-unwind-non-canonical=true %t/my-personality.s -o %t/arm64-32-my-personality.o
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-watchos -emit-compact-unwind-non-canonical=true %t/main.s -o %t/arm64-32-main.o
# RUN: %lld-watchos -lSystem -lc++ %t/arm64-32-my-personality.o %t/arm64-32-main.o -o %t/arm64-32-personality-first
# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --rebase %t/arm64-32-personality-first | FileCheck %s --check-prefixes=FIRST,CHECK -D#%x,BASE=0x4000 -DSEG=__TEXT
# RUN: %lld-watchos -dead_strip -lSystem -lc++ %t/arm64-32-main.o %t/arm64-32-my-personality.o -o %t/arm64-32-personality-second
# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --rebase %t/arm64-32-personality-second | FileCheck %s --check-prefixes=SECOND,CHECK -D#%x,BASE=0x4000 -DSEG=__TEXT

# RUN: %lld -arch x86_64 -rename_section __TEXT __gcc_except_tab __RODATA __gcc_except_tab -lSystem -lc++ %t/x86_64-my-personality.o %t/x86_64-main.o -o %t/x86_64-personality-first
# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --rebase %t/x86_64-personality-first | FileCheck %s --check-prefixes=FIRST,CHECK -D#%x,BASE=0x100000000 -DSEG=__RODATA
# RUN: %lld -dead_strip -arch x86_64 -rename_section __TEXT __gcc_except_tab __RODATA __gcc_except_tab -lSystem -lc++ %t/x86_64-main.o %t/x86_64-my-personality.o -o %t/x86_64-personality-second
# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --rebase %t/x86_64-personality-second | FileCheck %s --check-prefixes=SECOND,CHECK -D#%x,BASE=0x100000000 -DSEG=__RODATA

# FIRST:      Indirect symbols for (__DATA_CONST,__got)
# FIRST-NEXT: address                    index name
# FIRST-DAG:  0x[[#%x,GXX_PERSONALITY:]] [[#]] ___gxx_personality_v0
# FIRST-DAG:  0x[[#%x,MY_PERSONALITY:]] LOCAL

# SECOND:      Indirect symbols for (__DATA_CONST,__got)
# SECOND-NEXT: address                    index name
# SECOND-DAG:  0x[[#%x,GXX_PERSONALITY:]] [[#]] ___gxx_personality_v0
# SECOND-DAG:  0x[[#%x,MY_PERSONALITY:]] LOCAL

# CHECK:      SYMBOL TABLE:
# CHECK-DAG:  [[#%x,MAIN:]]       g  F __TEXT,__text _main
# CHECK-DAG:  [[#%x,QUUX:]]       g  F __TEXT,__text _quux
# CHECK-DAG:  [[#%x,FOO:]]        l  F __TEXT,__text _foo
# CHECK-DAG:  [[#%x,BAZ:]]        l  F __TEXT,__text _baz
# CHECK-DAG:  [[#%x,EXCEPTION0:]] g  O [[SEG]],__gcc_except_tab _exception0
# CHECK-DAG:  [[#%x,EXCEPTION1:]] g  O [[SEG]],__gcc_except_tab _exception1

# CHECK:      Contents of __unwind_info section:
# CHECK:        Personality functions: (count = 2)
# CHECK-DAG:     personality[{{[0-9]+}}]: 0x{{0*}}[[#MY_PERSONALITY-BASE]]
# CHECK-DAG:     personality[{{[0-9]+}}]: 0x{{0*}}[[#GXX_PERSONALITY-BASE]]
# CHECK:      Top level indices: (count = 2)
# CHECK-DAG:    [0]: function offset={{.*}}, 2nd level page offset=0x[[#%x,PAGEOFF:]],
# CHECK-DAG:    [1]: function offset={{.*}}, 2nd level page offset=0x00000000,
# CHECK:        LSDA descriptors:
# CHECK-DAG:     function offset=0x[[#%.8x,FOO-BASE]],  LSDA offset=0x[[#%.8x,EXCEPTION0-BASE]]
# CHECK-DAG:     function offset=0x[[#%.8x,MAIN-BASE]], LSDA offset=0x[[#%.8x,EXCEPTION1-BASE]]
# CHECK:        Second level indices:
# CHECK-NEXT:   Second level index[0]: offset in section=0x[[#%.8x,PAGEOFF]]
# CHECK-DAG:     function offset=0x[[#%.8x,MAIN-BASE]], encoding
# CHECK-DAG:     function offset=0x[[#%.8x,FOO-BASE]], encoding
# CHECK-DAG:     function offset=0x[[#%.8x,BAZ-BASE]], encoding
# CHECK-DAG:     function offset=0x[[#%.8x,QUUX-BASE]], encoding{{.*}}=0x00000000

## Check that we do not add rebase opcodes to the compact unwind section.
# CHECK:      Rebase table:
# CHECK-NEXT: segment      section        address          type
# CHECK-NEXT: __DATA_CONST __got          0x{{[0-9A-F]*}}  pointer
# CHECK-NOT:  __TEXT

## Check that we don't create an __unwind_info section if no unwind info
## remains after dead-stripping.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 \
# RUN:   %t/empty-after-dead-strip.s -o %t/x86_64-empty-after-dead-strip.o
# RUN: %lld -dylib -dead_strip -arch x86_64 -lSystem \
# RUN:   %t/x86_64-empty-after-dead-strip.o -o %t/x86_64-empty-after-strip.dylib
# RUN: llvm-objdump --macho --unwind-info %t/x86_64-empty-after-strip.dylib | \
# RUN:   FileCheck %s --check-prefixes=NOUNWIND --allow-empty
# NOUNWIND-NOT: Contents of __unwind_info section:

#--- my-personality.s
.globl _my_personality, _exception0
.text
.p2align 2
.no_dead_strip _foo
_foo:
  .cfi_startproc
## This will generate a section relocation.
  .cfi_personality 155, _my_personality
  .cfi_lsda 16, _exception0
  .cfi_def_cfa_offset 16
  ret
  .cfi_endproc

.p2align 2
.no_dead_strip _bar
_bar:
  .cfi_startproc
## Check that we dedup references to the same statically-linked personality.
  .cfi_personality 155, _my_personality
  .cfi_lsda 16, _exception0
  .cfi_def_cfa_offset 16
  ret
  .cfi_endproc

.data
.p2align 2
## We put this personality in `__data` to test if we correctly handle
## personality symbols whose output addresses occur after that of the
## `__unwind_info` section.
_my_personality:
  ret

.section __TEXT,__gcc_except_tab
_exception0:
  .space 1

.subsections_via_symbols

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
.no_dead_strip _quux
_quux:
  ret

.globl _abs
.no_dead_strip _abs
_abs = 4

.p2align 2
.no_dead_strip _baz
_baz:
  .cfi_startproc
## This will generate a symbol relocation. Check that we reuse the personality
## referenced by the section relocation in my_personality.s.
  .cfi_personality 155, _my_personality
  .cfi_lsda 16, _exception1
  .cfi_def_cfa_offset 16
  ret
  .cfi_endproc

.globl _stripped
_stripped:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_lsda 16, _exception1
  .cfi_def_cfa_offset 16
  ret
  .cfi_endproc


.section __TEXT,__gcc_except_tab
_exception1:
  .space 1

.subsections_via_symbols

#--- empty-after-dead-strip.s
.text

## Local symbol with unwind info.
## The symbol is removed by -dead_strip.
_foo :
  .cfi_startproc
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

.subsections_via_symbols
