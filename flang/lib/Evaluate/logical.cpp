//===-- lib/Evaluate/logical.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/logical.h"

namespace Fortran::evaluate::value {

template class Logical<8>;
template class Logical<16>;
template class Logical<32>;
template class Logical<64>;
} // namespace Fortran::evaluate::value
