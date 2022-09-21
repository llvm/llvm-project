//===-- Definition of macros from sys/mman.h ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_LINUX_SYS_MMAN_MACROS_H
#define __LLVM_LIBC_MACROS_LINUX_SYS_MMAN_MACROS_H

// Memory protection flags. (mmap, munmap, mprotect)
#define PROT_NONE 0x0
#define PROT_READ 0x1
#define PROT_WRITE 0x2
#define PROT_EXEC 0x4
// These protection flags are only valid for mprotect.
#define PROT_GROWSUP 0x1000000
#define PROT_GROWSDOWN 0x2000000

// Memory mapping flags. (mmap, munmap)
#define MAP_FAILED ((void *)-1)
// These are the sharing types, and exactly one of these must be set.
#define MAP_FILE 0x0 // Compatibility flag. Ignored.
#define MAP_SHARED 0x1
#define MAP_PRIVATE 0x2
#define MAP_SHARED_VALIDATE 0x3
// 0x4-0xf are unused.
// These are the other flags, and zero or more of these can be set.
#define MAP_FIXED 0x10
#define MAP_ANONYMOUS 0x20
#define MAP_ANON MAP_ANONYMOUS
#define MAP_32BIT 0x40
// 0x80 is unused.
#define MAP_GROWSDOWN 0x100
#define MAP_DENYWRITE 0x800
#define MAP_EXECUTABLE 0x1000 // Compatibility flag. Ignored.
#define MAP_LOCKED 0x2000
#define MAP_NORESERVE 0x4000
#define MAP_POPULATE 0x8000
#define MAP_NONBLOCK 0x10000
#define MAP_STACK 0x20000
#define MAP_HUGETLB 0x40000
#define MAP_SYNC 0x80000
#define MAP_FIXED_NOREPLACE 0x100000
// HUGETLB support macros. If ma_HUGHTLB is set, the bits under MAP_HUGE_MASK
// represent the base-2 logarithm of the desired page size, and MAP_HUGE_2MB/1GB
// are common sizes.
#define MAP_HUGE_SHIFT 26
#define MAP_HUGE_MASK 0x3f
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)

// Memory sync flags. (msync)
#define MS_ASYNC 1
#define MS_INVALIDATE 2
#define MS_SYNC 4

// Memory advice flags. (madvise)
#define MADV_NORMAL 0
#define MADV_RANDOM 1
#define MADV_SEQUENTIAL 2
#define MADV_WILLNEED 3
#define MADV_DONTNEED 4
#define MADV_FREE 8
#define MADV_REMOVE 9
#define MADV_DONTFORK 10
#define MADV_DOFORK 11
#define MADV_MERGEABLE 12
#define MADV_UNMERGEABLE 13
#define MADV_HUGEPAGE 14
#define MADV_NOHUGEPAGE 15
#define MADV_DONTDUMP 16
#define MADV_DODUMP 17
#define MADV_WIPEONFORK 18
#define MADV_KEEPONFORK 19
#define MADV_COLD 20
#define MADV_PAGEOUT 21
#define MADV_HWPOISON 100

// Posix memory advise flags. (posix_madvise)
#define POSIX_MADV_NORMAL MADV_NORMAL
#define POSIX_MADV_SEQUENTIAL MADV_SEQUENTIAL
#define POSIX_MADV_RANDOM MADV_RANDOM
#define POSIX_MADV_WILLNEED MADV_WILLNEED
#define POSIX_MADV_DONTNEED MADV_DONTNEED

#endif // __LLVM_LIBC_MACROS_LINUX_SYS_MMAN_MACROS_H
