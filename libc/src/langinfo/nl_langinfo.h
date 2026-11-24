//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for nl_langinfo.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_LANGINFO_NL_LANGINFO_H
#define LLVM_LIBC_SRC_LANGINFO_NL_LANGINFO_H

#include "hdr/types/nl_item.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

char *nl_langinfo(nl_item item);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_LANGINFO_NL_LANGINFO_H
