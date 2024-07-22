# REQUIRES: aarch64

## Old versions of Android (Android 11 & 12) have very strict parsing logic on
## the layout of the ELF note. This test ensures that backwards compatibility is
## maintained, i.e. new versions of the linker will still produce binaries that
## can be run on these versions of Android.

# RUN: llvm-mc --filetype=obj -triple=aarch64-none-linux-android %s -o %t.o
# RUN: ld.lld -shared --android-memtag-mode=async --android-memtag-heap %t.o -o %t
# RUN: llvm-readelf --memtag %t | FileCheck %s --check-prefixes=CHECK,HEAP,NOSTACK,ASYNC

# RUN: ld.lld -shared --android-memtag-mode=sync --android-memtag-heap %t.o -o %t
# RUN: llvm-readelf --memtag %t | FileCheck %s --check-prefixes=CHECK,HEAP,NOSTACK,SYNC

# RUN: ld.lld -shared --android-memtag-mode=async --android-memtag-stack %t.o -o %t
# RUN: llvm-readelf --memtag %t | FileCheck %s --check-prefixes=CHECK,NOHEAP,STACK,ASYNC

# RUN: ld.lld -shared --android-memtag-mode=sync --android-memtag-stack %t.o -o %t
# RUN: llvm-readelf --memtag %t | FileCheck %s --check-prefixes=CHECK,NOHEAP,STACK,SYNC

# RUN: ld.lld -shared --android-memtag-mode=async --android-memtag-heap \
# RUN:    --android-memtag-stack %t.o -o %t
# RUN: llvm-readelf --memtag %t | FileCheck %s --check-prefixes=CHECK,HEAP,STACK,ASYNC

# RUN: ld.lld -shared --android-memtag-mode=sync --android-memtag-heap \
# RUN:    --android-memtag-stack %t.o -o %t
# RUN: llvm-readelf --memtag %t | FileCheck %s --check-prefixes=CHECK,HEAP,STACK,SYNC

# RUN: ld.lld -shared --android-memtag-heap %t.o -o %t 2>&1 | \
# RUN:    FileCheck %s --check-prefixes=MISSING-MODE
# RUN: ld.lld -shared --android-memtag-stack %t.o -o %t 2>&1 | \
# RUN:    FileCheck %s --check-prefixes=MISSING-MODE
# RUN: ld.lld -shared --android-memtag-heap --android-memtag-stack %t.o -o %t 2>&1 | \
# RUN:    FileCheck %s --check-prefixes=MISSING-MODE
# MISSING-MODE: warning: --android-memtag-mode is unspecified, leaving
# MISSING-MODE: --android-memtag-{{(heap|stack)}} a no-op

# CHECK: Memtag Dynamic Entries
# SYNC:    AARCH64_MEMTAG_MODE: Synchronous (0)
# ASYNC:   AARCH64_MEMTAG_MODE: Asynchronous (1)
# HEAP:    AARCH64_MEMTAG_HEAP: Enabled (1)
# NOHEAP:  AARCH64_MEMTAG_HEAP: Disabled (0)
# STACK:   AARCH64_MEMTAG_STACK: Enabled (1)
# NOSTACK: AARCH64_MEMTAG_STACK: Disabled (0)

# CHECK:       Memtag Android Note
# ASYNC-NEXT:  Tagging Mode: ASYNC
# SYNC-NEXT:   Tagging Mode: SYNC
# HEAP-NEXT:   Heap: Enabled
# NOHEAP-NEXT: Heap: Disabled
# STACK-NEXT:   Stack: Enabled
# NOSTACK-NEXT: Stack: Disabled

# RUN: not ld.lld -shared --android-memtag-mode=asymm --android-memtag-heap 2>&1 | \
# RUN:    FileCheck %s --check-prefix=BAD-MODE
# BAD-MODE: error: unknown --android-memtag-mode value: "asymm", should be one of
# BAD-MODE: {async, sync, none}

# RUN: ld.lld -static --android-memtag-mode=sync --android-memtag-heap \
# RUN:    --android-memtag-stack %t.o -o %t
# RUN: llvm-readelf --memtag %t | FileCheck %s --check-prefixes=STATIC

# STATIC:      Memtag Dynamic Entries:
# STATIC-NEXT: < none found >
# STATIC:      Memtag Android Note:
# STATIC-NEXT:  Tagging Mode: SYNC
# STATIC-NEXT:  Heap: Enabled
# STATIC-NEXT:  Stack: Enabled


.globl _start
_start:
  ret
