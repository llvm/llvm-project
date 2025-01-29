//===-- Unittests for nearbyintf ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NearbyIntTest.h"

#include "src/math/nearbyintf.h"

LIST_NEARBYINT_TESTS(float, LIBC_NAMESPACE::nearbyintf)
