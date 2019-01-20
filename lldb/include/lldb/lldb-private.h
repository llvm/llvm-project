//===-- lldb-private.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_lldb_private_h_
#define lldb_lldb_private_h_

#if defined(__cplusplus)

#include "lldb/lldb-private-defines.h"
#include "lldb/lldb-private-enumerations.h"
#include "lldb/lldb-private-interfaces.h"
#include "lldb/lldb-private-types.h"
#include "lldb/lldb-public.h"

namespace lldb_private {

const char *GetVersion();

} // namespace lldb_private

#endif // defined(__cplusplus)

#endif // lldb_lldb_private_h_
