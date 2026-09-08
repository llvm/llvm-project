## Recover an AArch64 glibc-style entry which has no FDE or symbol-table entry.
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-linux %s -o %t.o
# RUN: ld.lld --export-dynamic %t.o -o %t.exe
# RUN: llvm-strip --strip-all %t.exe -o %t.stripped
# RUN: llvm-bolt %t.stripped -o %t.bolt --allow-stripped \
# RUN:   --recover-relocations --print-relocation-recovery --print-only=_start \
# RUN:   2>&1 | FileCheck %s
# RUN: llvm-readelf -h %t.bolt | FileCheck %s --check-prefix=ELF

# CHECK: BOLT-INFO: recovered stripped entry at
# CHECK: Binary Function "_start"
# CHECK: Secondary Entry Points : __ENTRY__start
# CHECK: adrp x0, __ENTRY_
# CHECK-NEXT: add x0, x0, :lo12:__ENTRY_
# ELF: Entry point address: 0x{{[1-9a-fA-F][0-9a-fA-F]*}}

.text
.globl _start
.hidden _start
.type _start, %function
_start:
  adrp x0, .Lmain_wrapper
  add x0, x0, :lo12:.Lmain_wrapper
  bl __libc_start_main
  bl abort
.Lmain_wrapper:
  b main
.size _start, .-_start

.globl __libc_start_main
.type __libc_start_main, %function
__libc_start_main:
.cfi_startproc
  ret
.cfi_endproc
.size __libc_start_main, .-__libc_start_main

.globl abort
.type abort, %function
abort:
.cfi_startproc
  ret
.cfi_endproc
.size abort, .-abort

.globl main
.type main, %function
main:
.cfi_startproc
  ret
.cfi_endproc
.size main, .-main
