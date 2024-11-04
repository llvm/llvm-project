//===---------- Baremetal implementation of IO utils ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "io.h"

#include "src/__support/CPP/string_view.h"

// This is intended to be provided by the vendor.
extern "C" void __llvm_libc_log_write(const char *msg, size_t len);

namespace LIBC_NAMESPACE {

void write_to_stderr(cpp::string_view msg) {
  __llvm_libc_log_write(msg.data(), msg.size());
}

} // namespace LIBC_NAMESPACE
