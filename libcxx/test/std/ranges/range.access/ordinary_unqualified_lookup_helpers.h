//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ACCESS_ORDINARY_UNQUALIFIED_LOOKUP_HELPERS_H
#define TEST_STD_RANGES_RANGE_ACCESS_ORDINARY_UNQUALIFIED_LOOKUP_HELPERS_H

namespace ordinary_unqualified_lookup_helpers {
struct StructWithGlobalRangeAccessFunctions {};
} // namespace ordinary_unqualified_lookup_helpers

int* begin(ordinary_unqualified_lookup_helpers::StructWithGlobalRangeAccessFunctions);
int* end(ordinary_unqualified_lookup_helpers::StructWithGlobalRangeAccessFunctions);
int* rbegin(ordinary_unqualified_lookup_helpers::StructWithGlobalRangeAccessFunctions);
int* rend(ordinary_unqualified_lookup_helpers::StructWithGlobalRangeAccessFunctions);
unsigned int size(ordinary_unqualified_lookup_helpers::StructWithGlobalRangeAccessFunctions);

#endif // TEST_STD_RANGES_RANGE_ACCESS_ORDINARY_UNQUALIFIED_LOOKUP_HELPERS_H
