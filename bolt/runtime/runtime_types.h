//===- bolt/runtime/runtime_types.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_RUNTIME_TYPES
#define LLVM_TOOLS_LLVM_BOLT_RUNTIME_TYPES

// FIXME: cleanup here

#if defined(__linux__)

#include <cstddef>
#include <cstdint>

#include "config.h"

#ifdef HAVE_ELF_H
#include <elf.h>
#endif

#elif defined(__APPLE__)

typedef __SIZE_TYPE__ size_t;
#define __SSIZE_TYPE__                                                         \
  __typeof__(_Generic((__SIZE_TYPE__)0,                                        \
                 unsigned long long int: (long long int)0,                     \
                 unsigned long int: (long int)0,                               \
                 unsigned int: (int)0,                                         \
                 unsigned short: (short)0,                                     \
                 unsigned char: (signed char)0))
typedef __SSIZE_TYPE__ ssize_t;

typedef unsigned long long uint64_t;
typedef unsigned uint32_t;
typedef unsigned char uint8_t;

typedef long long int64_t;
typedef int int32_t;

#else
#error "For Linux or MacOS only"
#endif

#ifndef __LP64__
#error "Only LP64 data model is supported"
#endif

#define AT_FDCWD -100

#define PROT_READ 0x1  /* Page can be read.  */
#define PROT_WRITE 0x2 /* Page can be written.  */
#define PROT_EXEC 0x4  /* Page can be executed.  */
#define PROT_NONE 0x0  /* Page can not be accessed.  */
#define PROT_GROWSDOWN                                                         \
  0x01000000 /* Extend change to start of                                      \
                growsdown vma (mprotect only).  */
#define PROT_GROWSUP                                                           \
  0x02000000 /* Extend change to start of                                      \
                growsup vma (mprotect only).  */

/* Sharing types (must choose one and only one of these).  */
#define MAP_SHARED 0x01  /* Share changes.  */
#define MAP_PRIVATE 0x02 /* Changes are private.  */
#define MAP_FIXED 0x10   /* Interpret addr exactly.  */

#if defined(__APPLE__)
#define MAP_ANONYMOUS 0x1000
#else
#define MAP_ANONYMOUS 0x20
#endif

#define MADV_HUGEPAGE 14 /* Worth backing with hugepages */

/* set the state of the "THP disable" flags for the calling thread */
#define PR_SET_THP_DISABLE 41

#define SEEK_SET 0 /* Seek from beginning of file.  */
#define SEEK_CUR 1 /* Seek from current position.  */
#define SEEK_END 2 /* Seek from end of file.  */

#define O_RDONLY 0
#define O_WRONLY 1
#define O_RDWR 2
#define O_CREAT 64
#define O_TRUNC 512
#define O_APPEND 1024
#define O_CLOEXEC 524288

#define SIG_BLOCK 0
#define SIG_UNBLOCK 1
#define SIG_SETMASK 2

#define CLONE_CHILD_CLEARTID 0x00200000 /* clear the TID in the child */
#define CLONE_CHILD_SETTID 0x01000000   /* set the TID in the child */

#define SIGCHLD 17

/* Length of the entries in `struct utsname' is 65.  */
#define _UTSNAME_LENGTH 65

#define _NSIG 64
#define _NSIG_BPW 64
#define _NSIG_WORDS (_NSIG / _NSIG_BPW)

// x86_64, aarch64, riscv64 / Linux/Unix - LP64
typedef long int ssize_t;
typedef long int off_t;
typedef int pid_t;
typedef unsigned mode_t;

// Anonymous namespace covering everything but our library entry point
namespace {

struct dirent64 {
  uint64_t d_ino;          /* Inode number */
  int64_t d_off;           /* Offset to next linux_dirent */
  unsigned short d_reclen; /* Length of this linux_dirent */
  unsigned char d_type;
  char d_name[]; /* Filename (null-terminated) */
                 /* length is actually (d_reclen - 2 -
                   offsetof(struct linux_dirent, d_name)) */
};

struct UtsNameTy {
  char sysname[_UTSNAME_LENGTH];  /* Operating system name (e.g., "Linux") */
  char nodename[_UTSNAME_LENGTH]; /* Name within "some implementation-defined
                      network" */
  char release[_UTSNAME_LENGTH]; /* Operating system release (e.g., "2.6.28") */
  char version[_UTSNAME_LENGTH]; /* Operating system version */
  char machine[_UTSNAME_LENGTH]; /* Hardware identifier */
  char domainname[_UTSNAME_LENGTH]; /* NIS or YP domain name */
};

struct timespec {
  uint64_t tv_sec;  /* seconds */
  uint64_t tv_nsec; /* nanoseconds */
};

typedef struct {
  unsigned long sig[_NSIG_WORDS];
} sigset_t;

} // anonymous namespace

#endif /* LLVM_TOOLS_LLVM_BOLT_RUNTIME_TYPES */
