//===-- HostInfoSwift.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_COMMON_HOSTINFOSWIFT_H
#define LLDB_HOST_COMMON_HOSTINFOSWIFT_H

#include "lldb/lldb-forward.h"
#include "llvm/ADT/Twine.h"

namespace lldb_private {
bool VerifySwiftPath(const llvm::Twine &swift_path);

bool DefaultComputeSwiftResourceDirectory(FileSpec &lldb_shlib_spec,
                                          FileSpec &file_spec, bool verify);
} // namespace lldb_private
#endif
