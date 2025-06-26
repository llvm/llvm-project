//===-- Implementation of wcstok ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/wcstok.h"

#include "hdr/types/wchar_t.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(wchar_t *, wcstok,
                   (wchar_t *__restrict str, const wchar_t *__restrict delim,
                    wchar_t **__restrict ptr)) {
    if (str == nullptr)
        str = *ptr;
    
    while (*str != L'\0') {
        bool inDelim = false;
        for (const wchar_t* delim_ptr = delim; delim_ptr != L'\0'; delim_ptr++) {
            
        }
    }
}

} // namespace LIBC_NAMESPACE_DECL
