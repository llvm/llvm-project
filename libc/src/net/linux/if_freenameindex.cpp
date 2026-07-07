//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of if_freenameindex.
///
//===----------------------------------------------------------------------===//

#include "src/net/if_freenameindex.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/struct_if_nameindex.h"
#include "src/__support/CPP/new.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, if_freenameindex, (struct if_nameindex * ptr)) {
  delete[] reinterpret_cast<uint8_t *>(ptr);
}

} // namespace LIBC_NAMESPACE_DECL
