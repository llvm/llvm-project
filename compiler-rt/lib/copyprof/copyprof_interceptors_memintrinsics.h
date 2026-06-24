//===-- copyprof_interceptors_memintrinsics.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file declares interceptors for memory intrinsic functions, ensuring
/// shadow memory consistency across block memory operations.
///
//===----------------------------------------------------------------------===//

#ifndef COPYPROF_INTERCEPTORS_MEMINTRINSICS_H
#define COPYPROF_INTERCEPTORS_MEMINTRINSICS_H

#include "copyprof_interface_internal.h"
#include "copyprof_internal.h"
#include "interception/interception.h"

namespace __copyprof {

#define COPYPROF_WRITE_RANGE(offset, size)   \
  do {                                       \
    __copyprof_store_callback(offset, size); \
  } while (0)

}  // namespace __copyprof

#endif  // COPYPROF_INTERCEPTORS_MEMINTRINSICS_H
