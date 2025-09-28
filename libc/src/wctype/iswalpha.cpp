//===-- Implementation of iswalpha ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wctype/iswalpha.h"
#include "src/__support/common.h"
#include "src/__support/wctype_utils.h"

#include "hdr/types/wint_t.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, iswalpha, (wint_t c)) { return internal::iswalpha(c); }

} // namespace LIBC_NAMESPACE_DECL
