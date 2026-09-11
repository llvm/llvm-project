## Check that non-fixed injected code is allocated immediately after split cold
## fragments in non-relocation mode and that LongJmp uses the same address.

# REQUIRES: system-linux, asserts, bolt-runtime, target=aarch64{{.*}}

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %t/input.s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib
# RUN: llvm-bolt %t.exe -o %t.bolt --data %t/profile.fdata --lite=0 --hugify \
# RUN:   --split-functions --split-all-cold --debug-only=longjmp \
# RUN:   2>&1 | tee %t.log
# RUN: llvm-nm --format=posix %t.bolt >> %t.log
# RUN: FileCheck %s < %t.log

# CHECK: BOLT-WARNING: non-relocation mode for AArch64 is not fully supported
# CHECK: BOLT-DEBUG: LongJmp layout: cold fragment _start ends at 0x[[INJECTED:[0-9a-f]+]]
# CHECK: BOLT-DEBUG: LongJmp layout: section .text.injected starts at 0x[[INJECTED]]
# CHECK: BOLT-DEBUG: LongJmp layout: main fragment __bolt_hugify_start_program starts at 0x[[INJECTED]]
# CHECK: __bolt_hugify_start_program t [[INJECTED]]

#--- input.s
  .text
  .globl _start
  .type _start, %function
_start:
  cbnz x0, .Lhot
.Lcold:
  sub x0, x0, #1
  ret
.Lhot:
  add x0, x0, #1
  ret
  .size _start, .-_start

#--- profile.fdata
no_lbr
1 _start 0 10
1 _start 4 0
1 _start c 10
