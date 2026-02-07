//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cxxabi.h>

// Make sure the `abi` namespace already exists
namespace abi_should_exist = abi;

// Make sure `abi` is an alias for `__cxxabiv1`
namespace abi = __cxxabiv1;
