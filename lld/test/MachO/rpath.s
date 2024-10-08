# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -o %t %t.o

## Check that -rpath generates LC_RPATH.
# RUN: %lld -o %t %t.o -rpath /some/rpath -rpath /another/rpath
# RUN: llvm-objdump --macho --all-headers %t | FileCheck %s
# CHECK:      LC_RPATH
# CHECK-NEXT: cmdsize 24
# CHECK-NEXT: path /some/rpath
# CHECK:      LC_RPATH
# CHECK-NEXT: cmdsize 32
# CHECK-NEXT: path /another/rpath

## Check that -rpath entries are deduplicated.
# RUN: not %lld %t.o -o /dev/null -rpath /some/rpath -rpath /other/rpath -rpath /some/rpath 2>&1 | \
# RUN:     FileCheck --check-prefix=FATAL %s
# FATAL: error: duplicate -rpath '/some/rpath' ignored [--warn-duplicate-rpath]

# RUN: %lld -o %t-dup %t.o -rpath /some/rpath -rpath /other/rpath -rpath /some/rpath --no-warn-duplicate-rpath
# RUN: llvm-objdump --macho --all-headers %t-dup | FileCheck %s --check-prefix=DEDUP
# DEDUP:      LC_RPATH
# DEDUP-NEXT: cmdsize 24
# DEDUP-NEXT: path /some/rpath
# DEDUP:      LC_RPATH
# DEDUP-NEXT: cmdsize 32
# DEDUP-NEXT: path /other/rpath
# DEDUP-NOT:  LC_RPATH

.text
.global _main
_main:
  mov $0, %rax
  ret
