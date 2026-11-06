# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o

## This aligns __big to end up on page 65534, which is still valid.
# RUN: %lld -dylib -fixup_chains -o %t %t.o -sectalign __DATA __big 0x10000000
# RUN: llvm-objdump --macho --chained-fixups %t | FileCheck %s
# CHECK:      chained starts in segment 1 (__DATA)
# CHECK:        page_count = 65535
# CHECK-NEXT:   page_start[0] = 0
# CHECK-NEXT:   page_start[1] = 65535 (DYLD_CHAINED_PTR_START_NONE)
# CHECK:        page_start[65534] = 0

## This aligns __big to end up on a page > UINT16_MAX.
# RUN: not %lld -dylib -fixup_chains -o %t.bad %t.o -sectalign __DATA __big 0x20000000 2>&1 \
# RUN:     | FileCheck --check-prefix=ERR %s
# ERR: error: page index in __DATA,__big too large for chained fixups; use -no_fixup_chains

.text
.space 0x1000

.data
.p2align 3
_p1:
  .quad _p1

.section __DATA,__big
.p2align 3
_p2:
  .quad _p2
