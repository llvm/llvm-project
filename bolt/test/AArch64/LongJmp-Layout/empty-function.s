## Check the two kinds of empty output handled by BinaryEmitter and mirrored by
## LongJmp. A zero-sized input function is not emitted at all. A function that
## becomes instruction-less is still emitted in relocation mode to define its
## main symbol, but is omitted when overwriting the original text in place.
## Structurally empty split fragments never produce symbols.

# REQUIRES: system-linux, asserts

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %t/empty-function.s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --lite=0 \
# RUN:   --pad-funcs-before=empty:64 --pad-funcs=empty:64 \
# RUN:   --debug-only=longjmp > %t.log 2>&1
# RUN: FileCheck %s --check-prefix=RELOC \
# RUN:   --implicit-check-not="LongJmp layout: main fragment empty" < %t.log
# RUN: llvm-nm %t.bolt | FileCheck %s --check-prefix=MAIN-SYMBOL
# RUN: %clang %cflags %t.o -o %t.noreloc.exe -nostdlib
# RUN: llvm-bolt %t.noreloc.exe -o %t.noreloc.bolt --lite=0 \
# RUN:   --pad-funcs-before=empty:64 --pad-funcs=empty:64 \
# RUN:   --debug-only=longjmp > %t.noreloc.log 2>&1
# RUN: FileCheck %s --check-prefix=NONRELOC \
# RUN:   --implicit-check-not="LongJmp layout: main fragment empty" \
# RUN:   --implicit-check-not="LongJmp layout: main fragment nop_only" \
# RUN:   < %t.noreloc.log

## --split-strategy=all deliberately leaves gaps in the fragment numbering
## when the secondary entry is forced back into the main fragment. This creates
## structurally empty fragments 1 and 2 followed by non-empty fragment 3.
## Compact code model bypasses the main/cold-only LongJmp layout path, allowing
## this subcase to check BinaryEmitter empty-fragment handling directly.
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %t/empty-fragments.s -o %t.fragments.o
# RUN: %clang %cflags %t.fragments.o -o %t.fragments.exe -nostdlib -Wl,-q
# RUN: llvm-bolt %t.fragments.exe -o %t.fragments.bolt --lite=0 \
# RUN:   --split-functions --split-strategy=all --compact-code-model
# RUN: llvm-nm %t.fragments.bolt | FileCheck %s --check-prefix=FRAGMENTS \
# RUN:   --implicit-check-not="_start.cold.0" \
# RUN:   --implicit-check-not="_start.cold.1"

# RELOC: BOLT-DEBUG: LongJmp layout: section .text starts at 0x{{[0-9a-f]+}}, alignment 0x{{[0-9a-f]+}}, 3 fragments
# RELOC: BOLT-DEBUG: LongJmp layout: main fragment _start starts at 0x
# RELOC: BOLT-DEBUG: LongJmp layout: main fragment next starts at 0x
# RELOC: BOLT-DEBUG: LongJmp layout: main fragment nop_only starts at 0x
# RELOC: BOLT-DEBUG: LongJmp layout: main fragment nop_only ends at 0x

# MAIN-SYMBOL: T nop_only

# NONRELOC: BOLT-WARNING: non-relocation mode for AArch64 is not fully supported
# NONRELOC: BOLT-DEBUG: LongJmp layout: main fragment _start starts at 0x
# NONRELOC: BOLT-DEBUG: LongJmp layout: main fragment next starts at 0x

# FRAGMENTS: t _start.cold.2

#--- empty-function.s
  .text
  .globl _start
  .type _start, %function
_start:
  bl next
  ret
  .size _start, .-_start

  .globl next
  .type next, %function
next:
  ret
  .size next, .-next

  .globl nop_only
  .type nop_only, %function
nop_only:
  nop
  .size nop_only, .-nop_only

  .globl empty
  .type empty, %function
empty:
  .size empty, .-empty

  .reloc 0, R_AARCH64_NONE

#--- empty-fragments.s
  .text
  .globl _start
  .type _start, %function
_start:
  cbz x0, secondary_entry
  ret
  .globl secondary_entry
secondary_entry:
  add x0, x0, #1
  b .Llast
.Llast:
  ret
  .size _start, .-_start

  .reloc 0, R_AARCH64_NONE
