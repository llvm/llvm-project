# REQUIRES: x86
# UNSUPPORTED: zlib

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o --compress-debug-sections=zlib --compress-debug-sections=none -o /dev/null 2>&1 | count 0
# RUN: not ld.lld %t.o --compress-debug-sections=zlib -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --implicit-check-not=error:
# RUN: not ld.lld %t.o --compress-sections=foo=zlib -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=CHECK2 --implicit-check-not=error:

# CHECK: error: --compress-debug-sections: LLVM was not built with LLVM_ENABLE_ZLIB or did not find zlib at build time
# CHECK2: error: --compress-sections: LLVM was not built with LLVM_ENABLE_ZLIB or did not find zlib at build time

.globl _start
_start:
