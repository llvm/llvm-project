//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// priority_queue& operator=(const priority_queue&) = default;

// Validate whether the container can be copy-assigned with an ADL-hijacking operator&

#include <queue>

#include "test_macros.h"
#include "operator_hijacker.h"

void test() {
  std::priority_queue<operator_hijacker> pqo;
  std::priority_queue<operator_hijacker> pq;
  pq = pqo;
}
