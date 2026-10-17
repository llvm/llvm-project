//===----- andes_vector.h - Andes Vector definitions ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ANDES_VECTOR_H_
#define _ANDES_VECTOR_H_

#include "riscv_vector.h"

#pragma clang riscv intrinsic andes_vector

#define __riscv_intrinsic_xandesvbfhcvt 1
#define __riscv_intrinsic_xandesvdot 1
#define __riscv_intrinsic_xandesvpackfph 1
#define __riscv_intrinsic_xandesvsintload 1

#endif //_ANDES_VECTOR_H_
