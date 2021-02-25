# REQUIRES: x86
#
# This tests that we can reference symbols from a dylib,
# re-exported by a top-level tapi document, which itself is
# re-exported by another top-level tapi document.
#
# RUN: rm -rf %t; mkdir -p %t
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %s -o %t/test.o
# RUN: %lld -o %t/test -syslibroot %S/Inputs/iPhoneSimulator.sdk -lReexportSystem %t/test.o
# RUN: llvm-objdump %t/test --macho --bind %t/test | FileCheck %s

# CHECK: segment  section   address            type        addend  dylib               symbol
# CHECK: __DATA   __data    0x{{[0-9a-f]*}}    pointer     0       libReexportSystem __crashreporter_info__
# CHECK: __DATA   __data    0x{{[0-9a-f]*}}    pointer     0       libReexportSystem _cache_create

.text
.globl _main

_main:
  ret

.data
// This symbol is from libSystem, which is re-exported by libReexportSystem.
// Reference it here to verify that it is visible.
.quad __crashreporter_info__

// This symbol is from /usr/lib/system/libcache.dylib, which is re-exported in libSystem.
.quad _cache_create
