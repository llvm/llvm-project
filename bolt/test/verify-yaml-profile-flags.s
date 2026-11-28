## Check that BOLT rewrites lbr as brstack and merge-fdata merges both spellings.
## Use a placeholder hash so the profiles work on both architectures.

# REQUIRES: target-x86_64 || target-aarch64

# RUN: split-file %s %t
# RUN: %clang %cflags -c %t/main.s -o %t/main.o
# RUN: ld.lld -q %t/main.o -o %t/exe
# RUN: llvm-bolt %t/exe -o %t/out -data %t/old.yaml -w %t/out.yaml --profile-ignore-hash

# RUN: FileCheck %s < %t/out.yaml
# RUN: merge-fdata %t/old.yaml %t/new.yaml | FileCheck %s --check-prefix=MERGE

# CHECK: profile-flags: {{\[ *brstack *\]$}}
# CHECK: name: _start
# CHECK: exec: 2{{$}}

# MERGE: profile-flags: {{\[ *brstack *\]$}}
# MERGE: name: _start
# MERGE: exec: 5{{$}}

//--- main.s
  .text
  .globl _start
  .type _start,@function
  .reloc 0, BFD_RELOC_NONE      // AArch64 requires relocations
_start:
  ret
  .size _start, .-_start

//--- old.yaml
---
header:
  profile-version: 1
  binary-name: exe
  profile-flags: [lbr]
functions:
  - { name: _start, fid: 1, hash: 0x0, exec: 2, nblocks: 1 }

//--- new.yaml
---
header:
  profile-version: 1
  binary-name: exe
  profile-flags: [brstack]
functions:
  - { name: _start, fid: 1, hash: 0x0, exec: 3, nblocks: 1 }
