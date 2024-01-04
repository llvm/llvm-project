//===-- Classes to capture properites of linux applications -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_CONFIG_LINUX_APP_H
#define LLVM_LIBC_CONFIG_LINUX_APP_H

#include "src/__support/macros/properties/architectures.h"

#include <stdint.h>

namespace LIBC_NAMESPACE {

// Data structure to capture properties of the linux/ELF TLS image.
struct TLSImage {
  // The load address of the TLS.
  uintptr_t address;

  // The byte size of the TLS image consisting of both initialized and
  // uninitialized memory. In ELF executables, it is size of .tdata + size of
  // .tbss. Put in another way, it is the memsz field of the PT_TLS header.
  uintptr_t size;

  // The byte size of initialized memory in the TLS image. In ELF exectubles,
  // this is the size of .tdata. Put in another way, it is the filesz of the
  // PT_TLS header.
  uintptr_t init_size;

  // The alignment of the TLS layout. It assumed that the alignment
  // value is a power of 2.
  uintptr_t align;
};

#if defined(LIBC_TARGET_ARCH_IS_X86_64) ||                                     \
    defined(LIBC_TARGET_ARCH_IS_AARCH64) ||                                    \
    defined(LIBC_TARGET_ARCH_IS_ANY_RISCV)
// At the language level, argc is an int. But we use uint64_t as the x86_64
// ABI specifies it as an 8 byte value. Likewise, in the ARM64 ABI, arguments
// are usually passed in registers.  x0 is a doubleword register, so this is
// 64 bit for aarch64 as well.
typedef uintptr_t ArgcType;

// At the language level, argv is a char** value. However, we use uint64_t as
// ABIs specify the argv vector be an |argc| long array of 8-byte values.
typedef uintptr_t ArgVEntryType;

typedef uintptr_t EnvironType;
#else
#error "argc and argv types are not defined for the target platform."
#endif

// Linux manpage on `proc(5)` says that the aux vector is an array of
// unsigned long pairs.
// (see: https://man7.org/linux/man-pages/man5/proc.5.html)
using AuxEntryType = unsigned long;
// Using the naming convention from `proc(5)`.
// TODO: Would be nice to use the aux entry structure from elf.h when available.
struct AuxEntry {
  AuxEntryType id;
  AuxEntryType value;
};

struct Args {
  ArgcType argc;

  // A flexible length array would be more suitable here, but C++ doesn't have
  // flexible arrays: P1039 proposes to fix this. So, for now we just fake it.
  // Even if argc is zero, "argv[argc] shall be a null pointer"
  // (ISO C 5.1.2.2.1) so one is fine. Also, length of 1 is not really wrong as
  // |argc| is guaranteed to be atleast 1, and there is an 8-byte null entry at
  // the end of the argv array.
  ArgVEntryType argv[1];
};

// Data structure which captures properties of a linux application.
struct AppProperties {
  // Page size used for the application.
  uintptr_t page_size;

  Args *args;

  // The properties of an application's TLS image.
  TLSImage tls;

  // Environment data.
  EnvironType *env_ptr;

  // Auxiliary vector data.
  AuxEntry *auxv_ptr;
};

extern AppProperties app;

// The descriptor of a thread's TLS area.
struct TLSDescriptor {
  // The size of the TLS area.
  uintptr_t size = 0;

  // The address of the TLS area. This address can be passed to cleanup
  // functions like munmap.
  uintptr_t addr = 0;

  // The value the thread pointer register should be initialized to.
  // Note that, dependending the target architecture ABI, it can be the
  // same as |addr| or something else.
  uintptr_t tp = 0;

  constexpr TLSDescriptor() = default;
};

// Create and initialize the TLS area for the current thread. Should not
// be called before app.tls has been initialized.
void init_tls(TLSDescriptor &tls);

// Cleanup the TLS area as described in |tls_descriptor|.
void cleanup_tls(uintptr_t tls_addr, uintptr_t tls_size);

// Set the thread pointer for the current thread.
bool set_thread_ptr(uintptr_t val);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_CONFIG_LINUX_APP_H
