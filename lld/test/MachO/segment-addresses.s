# REQUIRES: x86
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o
# RUN: %no-arg-lld -arch x86_64 -platform_version macos 11.0 11.0 \
# RUN:   -static -no_pie -pagezero_size 0 -image_base 0x100000 \
# RUN:   -segaddr __FIXED 0x400000 -o %t/out %t/test.o
# RUN: llvm-objdump --macho --private-headers %t/out | FileCheck %s
# RUN: not %no-arg-lld -arch x86_64 -platform_version macos 11.0 11.0 \
# RUN:   -static -image_base invalid -o /dev/null %t/test.o 2>&1 | \
# RUN:   FileCheck %s --check-prefix=BAD-BASE
# RUN: not %no-arg-lld -arch x86_64 -platform_version macos 11.0 11.0 \
# RUN:   -static -segaddr __FIXED 0x400001 -o /dev/null %t/test.o 2>&1 | \
# RUN:   FileCheck %s --check-prefix=BAD-SEGADDR

# CHECK:      segname __TEXT
# CHECK-NEXT: vmaddr 0x0000000000100000
# CHECK:      segname __DATA
# CHECK-NEXT: vmaddr 0x0000000000101000
# CHECK:      segname __FIXED
# CHECK-NEXT: vmaddr 0x0000000000400000
# CHECK:      segname __LINKEDIT
# CHECK-NEXT: vmaddr 0x0000000000401000

# BAD-BASE: error: -image_base: failed to parse 'invalid' as an address
# BAD-SEGADDR: error: -segaddr: address for segment __FIXED is not 4KiB aligned

.text
.globl _main
_main:
  retq

.data
.quad 0

.section __FIXED,__fixed
.quad 0
