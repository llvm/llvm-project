//===-- Implementation of cargf function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/complex/cargf.h"
#include "src/__support/common.h"
#include "src/__support/complex_type.h"
#include "src/math/atan2f.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, cargf, (_Complex float x)) {
    Complex<float> x_c = cpp::bit_cast<Complex<float>>(x);
    return atan2f(x_c.imag, x_c.real);
}

} // namespace LIBC_NAMESPACE_DECL
