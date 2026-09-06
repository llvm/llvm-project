# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/consumer.s -o %t/consumer.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/notlv.s -o %t/notlv.o

# RUN: %lld -arch arm64 -dylib -undefined dynamic_lookup -o %t/dylookup.dylib %t/consumer.o
# RUN: llvm-objdump --macho --section-headers --bind %t/dylookup.dylib | FileCheck %s

# RUN: %lld -arch arm64 -dylib -U _tlv -o %t/u.dylib %t/consumer.o
# RUN: llvm-objdump --macho --section-headers --bind %t/u.dylib | FileCheck %s

# RUN: %lld -arch arm64 -dylib -flat_namespace -undefined suppress -o %t/flat.dylib %t/consumer.o
# RUN: llvm-objdump --macho --section-headers --bind %t/flat.dylib | FileCheck %s

# CHECK:      __thread_ptrs
# CHECK-LABEL: Bind table:
# CHECK:      __DATA __thread_ptrs 0x{{[0-9a-f]+}} pointer 0 flat-namespace _tlv

## The import gets the flat-lookup ordinal.
# RUN: %lld -arch arm64 -dylib -undefined dynamic_lookup -fixup_chains \
# RUN:   -o %t/chained.dylib %t/consumer.o
# RUN: llvm-objdump --macho --chained-fixups %t/chained.dylib | FileCheck %s --check-prefix=CHAINED
# CHAINED: lib_ordinal = -2 (flat-namespace)

## A dylib that does define the symbol still tells us its thread-locality, so
## the mismatch check keeps working there.
# RUN: %lld -arch arm64 -dylib -install_name @rpath/libnotlv.dylib -o %t/libnotlv.dylib %t/notlv.o
# RUN: not %lld -arch arm64 -dylib -o /dev/null %t/consumer.o %t/libnotlv.dylib 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR
# ERR: error: {{.*}}TLVP_LOAD_PAGE21 relocation requires that symbol _tlv be thread-local

#--- consumer.s
.globl _readTlv
.p2align 2
_readTlv:
  adrp x8, _tlv@TLVPPAGE
  ldr  x8, [x8, _tlv@TLVPPAGEOFF]
  ret

#--- notlv.s
.globl _tlv
.data
_tlv:
  .quad 0
