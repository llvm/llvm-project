# REQUIRES: aarch64

## Old versions of Android (Android 11 & 12) have very strict parsing logic on
## the layout of the ELF note. This test ensures that backwards compatibility is
## maintained, i.e. new versions of the linker will still produce binaries that
## can be run on these versions of Android.

# RUN: llvm-mc --filetype=obj -triple=aarch64-none-linux-android %s -o %t.o
# RUN: ld.lld --android-memtag-mode=async --android-memtag-heap %t.o -o %t
# RUN: llvm-readelf -n %t | FileCheck %s --check-prefixes=NOTE,HEAP,NOSTACK,ASYNC

# RUN: ld.lld --android-memtag-mode=sync --android-memtag-heap %t.o -o %t
# RUN: llvm-readelf -n %t | FileCheck %s --check-prefixes=NOTE,HEAP,NOSTACK,SYNC

# RUN: ld.lld --android-memtag-mode=async --android-memtag-stack %t.o -o %t
# RUN: llvm-readelf -n %t | FileCheck %s --check-prefixes=NOTE,NOHEAP,STACK,ASYNC

# RUN: ld.lld --android-memtag-mode=sync --android-memtag-stack %t.o -o %t
# RUN: llvm-readelf -n %t | FileCheck %s --check-prefixes=NOTE,NOHEAP,STACK,SYNC

# RUN: ld.lld --android-memtag-mode=async --android-memtag-heap \
# RUN:    --android-memtag-stack %t.o -o %t
# RUN: llvm-readelf -n %t | FileCheck %s --check-prefixes=NOTE,HEAP,STACK,ASYNC

# RUN: ld.lld --android-memtag-mode=sync --android-memtag-heap \
# RUN:    --android-memtag-stack %t.o -o %t
# RUN: llvm-readelf -n %t | FileCheck %s --check-prefixes=NOTE,HEAP,STACK,SYNC

# RUN: ld.lld --android-memtag-heap %t.o -o %t
# RUN: llvm-readelf -n %t | FileCheck %s --check-prefixes=NOTE,HEAP,NOSTACK,SYNC

# RUN: ld.lld --android-memtag-stack %t.o -o %t
# RUN: llvm-readelf -n %t | FileCheck %s --check-prefixes=NOTE,NOHEAP,STACK,SYNC

# RUN: ld.lld --android-memtag-heap --android-memtag-stack %t.o -o %t
# RUN: llvm-readelf -n %t | FileCheck %s --check-prefixes=NOTE,HEAP,STACK,SYNC

# NOTE: .note.android.memtag
# NOTE-NEXT: Owner
# NOTE-NEXT: Android 0x00000004 NT_ANDROID_TYPE_MEMTAG (Android memory tagging
# NOTE-SAME: information)
# ASYNC-NEXT: Tagging Mode: ASYNC
# SYNC-NEXT: Tagging Mode: SYNC
# HEAP-NEXT: Heap: Enabled
# NOHEAP-NEXT: Heap: Disabled
## As of Android 12, stack MTE is unimplemented. However, we pre-emptively emit
## a bit that signifies to the dynamic loader to map the primary and thread
## stacks as PROT_MTE, in preparation for the bionic support.
# STACK-NEXT:   Stack: Enabled
# NOSTACK-NEXT: Stack: Disabled

# RUN: not ld.lld --android-memtag-mode=asymm --android-memtag-heap 2>&1 | \
# RUN:    FileCheck %s --check-prefix=BAD-MODE
# BAD-MODE: unknown --android-memtag-mode value: "asymm", should be one of {async, sync, none}

# RUN: not ld.lld --android-memtag-mode=async 2>&1 | \
# RUN:    FileCheck %s --check-prefix=MISSING-STACK-OR-HEAP
# MISSING-STACK-OR-HEAP: when using --android-memtag-mode, at least one of --android-memtag-heap or --android-memtag-stack is required

.globl _start
_start:
  ret
