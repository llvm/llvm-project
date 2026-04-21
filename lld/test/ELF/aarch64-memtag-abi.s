# REQUIRES: aarch64

# RUN: llvm-mc --filetype=obj -triple=aarch64-linux-gnu %s -o %t.o

# RUN: ld.lld -shared -z memtag-mode=async -z memtag-heap %t.o -o %t
# RUN: llvm-readelf --memtag %t | FileCheck %s --check-prefixes=CHECK,HEAP,NOSTACK,ASYNC

# RUN: ld.lld -shared -z memtag-mode=sync -z memtag-heap %t.o -o %t
# RUN: llvm-readelf --memtag %t | FileCheck %s --check-prefixes=CHECK,HEAP,NOSTACK,SYNC

# RUN: ld.lld -shared -z memtag-mode=async -z memtag-stack %t.o -o %t
# RUN: llvm-readelf --memtag %t | FileCheck %s --check-prefixes=CHECK,NOHEAP,STACK,ASYNC

# RUN: ld.lld -shared -z memtag-mode=sync -z memtag-stack %t.o -o %t
# RUN: llvm-readelf --memtag %t | FileCheck %s --check-prefixes=CHECK,NOHEAP,STACK,SYNC

# RUN: ld.lld -shared -z memtag-mode=async -z memtag-heap \
# RUN:    -z memtag-stack %t.o -o %t
# RUN: llvm-readelf --memtag %t | FileCheck %s --check-prefixes=CHECK,HEAP,STACK,ASYNC

# RUN: ld.lld -shared -z memtag-mode=sync -z memtag-heap \
# RUN:    -z memtag-stack %t.o -o %t
# RUN: llvm-readelf --memtag %t | FileCheck %s --check-prefixes=CHECK,HEAP,STACK,SYNC

# RUN: ld.lld -shared -z memtag-heap %t.o -o %t 2>&1 | \
# RUN:    FileCheck %s --check-prefixes=MISSING-MODE
# RUN: ld.lld -shared -z memtag-stack %t.o -o %t 2>&1 | \
# RUN:    FileCheck %s --check-prefixes=MISSING-MODE
# RUN: ld.lld -shared -z memtag-heap -z memtag-stack %t.o -o %t 2>&1 | \
# RUN:    FileCheck %s --check-prefixes=MISSING-MODE
# MISSING-MODE: warning: -z memtag-mode is none, leaving
# MISSING-MODE-SAME: -z memtag-{{(heap|stack)}} a no-op

# CHECK: Memtag Dynamic Entries
# SYNC:    AARCH64_MEMTAG_MODE: Synchronous (0)
# ASYNC:   AARCH64_MEMTAG_MODE: Asynchronous (1)
# HEAP:    AARCH64_MEMTAG_HEAP: Enabled (1)
# NOHEAP:  AARCH64_MEMTAG_HEAP: Disabled (0)
# STACK:   AARCH64_MEMTAG_STACK: Enabled (1)
# NOSTACK: AARCH64_MEMTAG_STACK: Disabled (0)

# CHECK-NOT: Memtag Android Note

# RUN: not ld.lld -shared -z memtag-mode=asymm -z memtag-heap 2>&1 | \
# RUN:    FileCheck %s --check-prefix=BAD-MODE
# BAD-MODE: error: unknown -z memtag-mode= value: asymm

# RUN: ld.lld -static -z memtag-mode=sync -z memtag-heap \
# RUN:    -z memtag-stack %t.o -o %t
# RUN: llvm-readelf --memtag %t | FileCheck %s --check-prefixes=STATIC

# STATIC:      Memtag Dynamic Entries:
# STATIC-NEXT: < none found >
# STATIC-NOT:  Memtag Android Note

.globl _start
_start:
  ret
