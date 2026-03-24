//===-- Scudo calloc ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "scudo_allocator.h"
#include "src/stdlib/calloc.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void *, calloc, (size_t nmemb, size_t size)) {
  scudo::uptr Product;
  if (UNLIKELY(scudo::checkForCallocOverflow(size, nmemb, &Product))) {
    if (Allocator.canReturnNull()) {
      scudo::LibcPAL::seterrno(ENOMEM);
      return nullptr;
    }
    scudo::reportCallocOverflow(nmemb, size);
  }

  void *Ptr =
      Allocator.allocate(Product, scudo::Chunk::Origin::Malloc,
                         static_cast<scudo::uptr>(SCUDO_MALLOC_ALIGNMENT), true);
  reportAllocation(Ptr, static_cast<size_t>(Product));
  return scudo::setErrnoOnNull(Ptr);
}

} // namespace LIBC_NAMESPACE_DECL
