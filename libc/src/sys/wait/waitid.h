//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for waitid.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_WAIT_WAITID_H
#define LLVM_LIBC_SRC_SYS_WAIT_WAITID_H

#include "hdr/types/id_t.h"
#include "hdr/types/idtype_t.h"
#include "hdr/types/siginfo_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int waitid(idtype_t idtype, id_t id, siginfo_t *infop, int options);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_WAIT_WAITID_H
