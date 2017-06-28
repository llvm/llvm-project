//===-- Support.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_LINUX_SUPPORT_H
#define LLDB_HOST_LINUX_SUPPORT_H

#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>

namespace lldb_private {

llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
getProcFile(::pid_t pid, ::pid_t tid, const llvm::Twine &file);

llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
getProcFile(::pid_t pid, const llvm::Twine &file);

llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
getProcFile(const llvm::Twine &file);

} // namespace lldb_private

#endif // #ifndef LLDB_HOST_LINUX_SUPPORT_H
