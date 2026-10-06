# REQUIRES: aarch64
## On arm64e, personality pointers are loaded from __auth_got, where dyld signs
## them with the IA key and address diversity: the unwinder authenticates them
## against the address of their slot. That applies to the personality array in
## __unwind_info and to a CIE's personality in __eh_frame. Other GOT references
## to the same symbol, `sym@GOT - .` and GOT loads, use an unsigned __got slot.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/pers.o %t/pers.s
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos \
# RUN:   -emit-compact-unwind-non-canonical=true -o %t/cu.o %t/cu.s
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos \
# RUN:   --emit-dwarf-unwind=always -o %t/dwarf.o %t/dwarf.s
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 -dylib \
# RUN:   -install_name @executable_path/libpers.dylib %t/pers.o -o %t/libpers.dylib
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 %t/cu.o %t/dwarf.o \
# RUN:   %t/libpers.dylib -o %t/test
# RUN: llvm-objdump --macho --section-headers --unwind-info %t/test > %t/out
# RUN: llvm-dwarfdump --eh-frame %t/test >> %t/out
# RUN: llvm-objdump --no-print-imm-hex --macho -d --no-show-raw-insn %t/test >> %t/out
# RUN: llvm-objdump --macho -s --section __TEXT,__const %t/test >> %t/out
# RUN: llvm-objdump --macho --dyld-info %t/test >> %t/out
# RUN: FileCheck %s -D#%x,BASE=0x100000000 < %t/out

## A personality defined in the same object is referenced from
## __compact_unwind through a section relocation. It gets an __auth_got slot
## of its own even though it already has an unsigned __got slot.
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos \
# RUN:   -emit-compact-unwind-non-canonical=true -o %t/local.o %t/local.s
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 %t/local.o \
# RUN:   -o %t/local
# RUN: llvm-objdump --macho --section-headers --syms --unwind-info \
# RUN:   %t/local > %t/local-out
# RUN: llvm-objdump --macho -s --section __TEXT,__const %t/local >> %t/local-out
# RUN: llvm-objdump --macho --dyld-info %t/local >> %t/local-out
# RUN: FileCheck %s --check-prefix=LOCAL -D#%x,BASE=0x100000000 < %t/local-out

# CHECK:      __const       {{[0-9a-f]+}} {{0*}}[[#%x,CONST:]] DATA
# CHECK:      __auth_got    00000008 {{0*}}[[#%x,AUTH_GOT:]] DATA
# CHECK-NEXT: __got         00000008 {{0*}}[[#%x,GOT:]] DATA

# CHECK:      Personality functions: (count = 1)
# CHECK-NEXT:   personality[1]: 0x{{0*}}[[#AUTH_GOT-BASE]]
# CHECK:      CIE
# CHECK:        Personality Address: {{0*}}[[#AUTH_GOT]]

# CHECK-LABEL: _main:
# CHECK:       adrp x0, {{.*}} ; 0x[[#%x,GOT_PAGE:]]
# CHECK-NEXT:  ldr x0, [x0, #[[#GOT-GOT_PAGE]]] ; literal pool symbol address: _pers

## `_pers@GOT - .`
# CHECK-LABEL: Contents of (__TEXT,__const) section
# CHECK-NEXT:  {{0*}}[[#CONST]] [[#%.8x,GOT-CONST]]
## An authenticated bind with key IA, diversity 0 and address diversity in
## __auth_got, and an unauthenticated bind in __got.
# CHECK-LABEL: dyld information:
# CHECK:       __DATA_CONST __auth_got 0x[[#%X,AUTH_GOT]] 0xC009000000000000 bind 0x0000 true IA 0x0 libpers _pers
# CHECK-NEXT:  __DATA_CONST __got 0x[[#%X,GOT]] 0x4000000000000000 bind 0x0 libpers _pers

# LOCAL:      __const       00000004 {{0*}}[[#%x,CONST:]] DATA
# LOCAL:      __auth_got    00000008 {{0*}}[[#%x,AUTH_GOT:]] DATA
# LOCAL-NEXT: __got         00000008 {{0*}}[[#%x,GOT:]] DATA
# LOCAL:      [[#%x,MY_PERS:]] g F __TEXT,__text _my_pers
# LOCAL:      Personality functions: (count = 1)
# LOCAL-NEXT:   personality[1]: 0x{{0*}}[[#AUTH_GOT-BASE]]
# LOCAL-LABEL: Contents of (__TEXT,__const) section
# LOCAL-NEXT:  {{0*}}[[#CONST]] [[#%.8x,GOT-CONST]]
## Both slots rebase _my_pers; only the __auth_got one is authenticated.
# LOCAL-LABEL: dyld information:
# LOCAL:       __DATA_CONST __auth_got 0x[[#%X,AUTH_GOT]] 0x{{[0-9A-F]+}} rebase 0x0000 true IA 0x[[#%X,MY_PERS]]
# LOCAL-NEXT:  __DATA_CONST __got 0x[[#%X,GOT]] 0x{{[0-9A-F]+}} rebase 0x[[#%X,MY_PERS]]

#--- pers.s
.globl _pers
.p2align 2
_pers:
  ret

#--- cu.s
.globl _main
.p2align 2
_main:
  .cfi_startproc
  .cfi_personality 155, _pers
  stp x29, x30, [sp, #-16]!
  mov x29, sp
  .cfi_def_cfa w29, 16
  .cfi_offset w30, -8
  .cfi_offset w29, -16
  bl _dwarf
  adrp x0, _pers@GOTPAGE
  ldr x0, [x0, _pers@GOTPAGEOFF]
  ldp x29, x30, [sp], #16
  ret
  .cfi_endproc

.section __TEXT,__const
.p2align 2
_pers_got_ref:
  .long _pers@GOT - .

#--- dwarf.s
.globl _dwarf
.p2align 2
_dwarf:
  .cfi_startproc
  .cfi_personality 155, _pers
  stp x29, x30, [sp, #-16]!
  .cfi_def_cfa_offset 16
  .cfi_offset w30, -8
  .cfi_offset w29, -16
  ldp x29, x30, [sp], #16
  ret
  .cfi_endproc

#--- local.s
.globl _main, _my_pers
.p2align 2
_main:
  .cfi_startproc
  .cfi_personality 155, _my_pers
  stp x29, x30, [sp, #-16]!
  mov x29, sp
  .cfi_def_cfa w29, 16
  .cfi_offset w30, -8
  .cfi_offset w29, -16
  ldp x29, x30, [sp], #16
  ret
  .cfi_endproc

.p2align 2
_my_pers:
  ret

.section __TEXT,__const
.p2align 2
_my_pers_got_ref:
  .long _my_pers@GOT - .

.subsections_via_symbols
