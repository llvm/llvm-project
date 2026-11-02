# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/ok.s -o %t/ok.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/straddle.s -o %t/straddle.o

## A pointer that ends exactly at a page boundary is fine.
# RUN: %lld -dylib -fixup_chains -o %t/ok %t/ok.o
# RUN: llvm-objdump --macho --chained-fixups %t/ok | FileCheck %s
# CHECK:      chained starts in segment 1 (__DATA)
# CHECK:        page_start[0] = 4088

## Chained fixups can't encode pointers straddling page boundaries.
# RUN: not %lld -dylib -fixup_chains -o %t/straddle %t/straddle.o 2>&1 \
# RUN:     | FileCheck --check-prefix=ERR %s
# ERR: error: {{.*}}straddle.o:(symbol _p+0x0): fixup straddles page boundary; stop packing structs containing pointers or link with -no_fixup_chains

## ...but it works without chained fixups.
# RUN: %lld -dylib -no_fixup_chains -o %t/straddle %t/straddle.o

#--- ok.s
.data
.p2align 12
.space 0xff8
_p:
  .quad _p

#--- straddle.s
.data
.p2align 12
.space 0xffc
_p:
  .quad _p
