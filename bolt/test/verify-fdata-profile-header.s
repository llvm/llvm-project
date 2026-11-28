## Check that BOLT reads legacy basic-sample profiles and that merge-fdata
## combines both header spellings into a profile with the canonical header.

# REQUIRES: target-x86_64 || target-aarch64

# RUN: split-file %s %t
# RUN: %clang %cflags -c %t/main.s -o %t/main.o
# RUN: ld.lld -q %t/main.o -o %t/exe
# RUN: llvm-bolt %t/exe -o %t/out -data %t/old.fdata | FileCheck %s
# RUN: merge-fdata %t/old.fdata %t/new.fdata | FileCheck %s --check-prefix=MERGE

# CHECK: BOLT-INFO: operating with basic samples profiling data (no brstack).
# CHECK: BOLT-INFO: 1 out of 1 functions in the binary (100.0%) have non-empty execution profile

# MERGE:      {{^}}no_brstack cycles:u instructions{{$}}
# MERGE-NEXT: {{^}}1 _start 0 5{{$}}

//--- main.s
  .text
  .globl _start
  .type _start,@function
  .reloc 0, BFD_RELOC_NONE      // AArch64 requires relocations
_start:
  nop
  ret
  .size _start, .-_start

//--- old.fdata
no_lbr cycles:u
1 _start 0 2

//--- new.fdata
no_brstack cycles:u instructions
1 _start 0 3
