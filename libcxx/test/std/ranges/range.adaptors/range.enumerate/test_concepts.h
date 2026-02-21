//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ENUMERATE_CONCEPTS_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ENUMERATE_CONCEPTS_H

template <class T>
concept HasMemberSize = requires(T t) { t.size(); };

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ENUMERATE_CONCEPTS_H
