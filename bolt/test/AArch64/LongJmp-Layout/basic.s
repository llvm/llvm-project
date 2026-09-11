## Check that LongJmp's layout matches actual emission for ordinary
## functions, function and basic-block alignment, explicit padding, and the
## optional function boundary markers.

# REQUIRES: system-linux, asserts

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.default.bolt --lite=0 \
# RUN:   --debug-only=longjmp > %t.default.log 2>&1
# RUN: llvm-nm -n --format=posix %t.default.bolt >> %t.default.log
# RUN: FileCheck %s --check-prefix=DEFAULT < %t.default.log
# RUN: llvm-bolt %t.exe -o %t.align.bolt --lite=0 \
# RUN:   --align-text=4096 --align-functions=256 \
# RUN:   --align-functions-max-bytes=255 --preserve-blocks-alignment \
# RUN:   --debug-only=longjmp > %t.align.log 2>&1
# RUN: llvm-nm -n --format=posix %t.align.bolt >> %t.align.log
# RUN: FileCheck %s --check-prefix=ALIGN < %t.align.log
# RUN: llvm-bolt %t.exe -o %t.max-align.bolt --lite=0 \
# RUN:   --align-functions=256 --align-functions-max-bytes=1 \
# RUN:   --debug-only=longjmp > %t.max-align.log 2>&1
# RUN: llvm-nm -n --format=posix %t.max-align.bolt >> %t.max-align.log
# RUN: FileCheck %s --check-prefix=MAX-ALIGN < %t.max-align.log
# RUN: llvm-bolt %t.exe -o %t.padding.bolt --lite=0 \
# RUN:   --pad-funcs-before=second:20 --pad-funcs=first:12 \
# RUN:   --break-funcs=first,second --mark-funcs \
# RUN:   --debug-only=longjmp > %t.padding.log 2>&1
# RUN: llvm-nm -n --format=posix %t.padding.bolt >> %t.padding.log
# RUN: FileCheck %s --check-prefix=PADDING < %t.padding.log
# RUN: %clang %cflags %t.o -o %t.noreloc.exe -nostdlib
# RUN: llvm-bolt %t.noreloc.exe -o %t.noreloc.bolt --lite=0 \
# RUN:   --preserve-blocks-alignment --debug-only=longjmp \
# RUN:   > %t.noreloc.log 2>&1
# RUN: FileCheck %s --check-prefix=NONRELOC < %t.noreloc.log

# DEFAULT: LongJmp layout: section .text starts at 0x[[TEXT:[0-9a-f]+]]
# DEFAULT: LongJmp layout: main fragment _start starts at 0x[[START:[0-9a-f]+]]
# DEFAULT: LongJmp layout: main fragment first starts at 0x[[FIRST:[0-9a-f]+]]
# DEFAULT: LongJmp layout: main fragment second starts at 0x[[SECOND:[0-9a-f]+]]
# DEFAULT: LongJmp layout: section .text ends at 0x
# DEFAULT: _start T [[START]]
# DEFAULT: first T [[FIRST]]
# DEFAULT: second T [[SECOND]]

# ALIGN: LongJmp layout: section .text starts at 0x[[TEXT:[0-9a-f]+]], alignment 0x1000
# ALIGN: LongJmp layout: main fragment _start starts at 0x[[START:[0-9a-f]+]]
# ALIGN: LongJmp layout: main fragment first starts at 0x[[FIRST:[0-9a-f]+]]
# ALIGN: LongJmp layout: basic block {{.*}} in first starts at 0x{{[0-9a-f]+}}
# ALIGN: LongJmp layout: main fragment second starts at 0x[[SECOND:[0-9a-f]+]]
# ALIGN: _start T [[START]]
# ALIGN: first T [[FIRST]]
# ALIGN: second T [[SECOND]]

# MAX-ALIGN: LongJmp layout: main fragment _start starts at 0x[[START:[0-9a-f]+]]
# MAX-ALIGN: LongJmp layout: main fragment first starts at 0x[[FIRST:[0-9a-f]+]]
# MAX-ALIGN: LongJmp layout: main fragment second starts at 0x[[SECOND:[0-9a-f]+]]
# MAX-ALIGN: _start T [[START]]
# MAX-ALIGN: first T [[FIRST]]
# MAX-ALIGN: second T [[SECOND]]

# PADDING: LongJmp layout: main fragment _start starts at 0x[[START:[0-9a-f]+]]
# PADDING: LongJmp layout: main fragment first starts at 0x[[FIRST:[0-9a-f]+]]
# PADDING: LongJmp layout: main fragment second starts at 0x[[SECOND:[0-9a-f]+]]
# PADDING: _start T [[START]]
# PADDING: first T [[FIRST]]
# PADDING: second T [[SECOND]]

# NONRELOC: BOLT-WARNING: non-relocation mode for AArch64 is not fully supported
# NONRELOC: BOLT-DEBUG: LongJmp layout starts at 0x
# NONRELOC: BOLT-DEBUG: LongJmp layout: main fragment first starts at 0x[[FIRST:[0-9a-f]+]]
# NONRELOC: BOLT-DEBUG: LongJmp layout: basic block {{.*}} in first starts at 0x[[FIRST]]
# NONRELOC: BOLT-DEBUG: LongJmp layout: basic block {{.*}} in first starts at 0x{{[0-9a-f]+}}
# NONRELOC: BOLT-DEBUG: LongJmp layout: basic block {{.*}} in first starts at 0x{{[0-9a-f]*[02468ace]0}}

  .text
  .p2align 6
  .globl _start
  .type _start, %function
_start:
  bl first
  bl second
  ret
  .size _start, .-_start

  .p2align 6
  .globl first
  .type first, %function
first:
  cbz x0, .Lfirst_aligned
  add x0, x0, #1
  ret
  .p2align 5
.Lfirst_aligned:
  sub x0, x0, #1
  ret
  .size first, .-first

  .p2align 7
  .globl second
  .type second, %function
second:
  add x1, x1, #1
  ret
  .size second, .-second

  .reloc 0, R_AARCH64_NONE
