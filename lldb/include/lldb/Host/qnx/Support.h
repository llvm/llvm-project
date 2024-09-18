//===-- Support.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_QNX_SUPPORT_H
#define LLDB_HOST_QNX_SUPPORT_H

#include <memory>

#include "lldb/Host/File.h"
#include "lldb/lldb-forward.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"

namespace lldb_private {

llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
getProcFile(::pid_t pid, const llvm::Twine &file);

llvm::Expected<lldb::FileUP>
openProcFile(::pid_t pid, const llvm::Twine &file, File::OpenOptions options,
             uint32_t permissions = lldb::eFilePermissionsFileDefault,
             bool should_close_fd = true);

} // namespace lldb_private

#endif // #ifndef LLDB_HOST_QNX_SUPPORT_H
