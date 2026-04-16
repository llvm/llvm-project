#ifndef LLVM_LIBC_SRC_WCHAR_WCSXFRM_H
#define LLVM_LIBC_SRC_WCHAR_WCSXFRM_H

#include "src/__support/macros/config.h"

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"

namespace LIBC_NAMESPACE_DECL {

size_t wcsxfrm(wchar_t *__restrict dest, const wchar_t *__restrict src,
               size_t n);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_WCHAR_WCSXFRM_H
