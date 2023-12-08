# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -dylib -fixup_chains -o %t %t.o
# RUN: llvm-objdump --macho --all-headers %t | FileCheck %s

## dyld always expects LC_DYLD_CHAINED_FIXUPS to point to a valid
## chained fixups header, even if there aren't any fixups.
# CHECK:            cmd LC_DYLD_CHAINED_FIXUPS
# CHECK-NEXT:   cmdsize 16
# CHECK-NEXT:   dataoff [[#]]
# CHECK-NEXT:  datasize 48

## While ld64 emits the root trie node even if there are no exports,
## setting the data size and offset to zero works too in practice.
# CHECK:            cmd LC_DYLD_EXPORTS_TRIE
# CHECK-NEXT:   cmdsize 16
# CHECK-NEXT:   dataoff 0
# CHECK-NEXT:  datasize 0

## Old load commands should not be generated.
# CHECK-NOT: cmd LC_DYLD_INFO_ONLY

_not_exported:
  .space 1
