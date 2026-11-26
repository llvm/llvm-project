//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for capset.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_CAPABILITY_CAPSET_H
#define LLVM_LIBC_SRC_SYS_CAPABILITY_CAPSET_H

#include "hdr/types/cap_user_data_t.h"
#include "hdr/types/cap_user_header_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int capset(cap_user_header_t hdrp, const cap_user_data_t datap);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_CAPABILITY_CAPSET_H
