//===-- Classes to capture properites of linux applications -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_CONFIG_LINUX_APP_H
#define LLVM_LIBC_CONFIG_LINUX_APP_H

#include <stdint.h>

namespace __llvm_libc {

// Data structure to capture properties of the linux/ELF TLS.
struct TLS {
  // The load address of the TLS.
  uintptr_t address;

  // The bytes size of the TLS.
  uintptr_t size;

  // The alignment of the TLS layout. It assumed that the alignment
  // value is a power of 2.
  uintptr_t align;
};

// Data structure which captures properties of a linux application.
struct AppProperties {
  // Page size used for the application.
  uintptr_t pageSize;

  // The properties of an application's TLS.
  TLS tls;

  // Environment data.
  uint64_t *envPtr;
};

extern AppProperties app;

// Creates and initializes the TLS area for the current thread. Should not
// be called before app.tls has been initialized.
void initTLS();

} // namespace __llvm_libc

#endif // LLVM_LIBC_CONFIG_LINUX_APP_H
