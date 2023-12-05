//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ACCESS_ORDINARY_UNQUALIFIED_LOOKUP_HELPERS_H
#define TEST_STD_RANGES_RANGE_ACCESS_ORDINARY_UNQUALIFIED_LOOKUP_HELPERS_H

// Note: this header should be included before any other header.
// Access functions defined here must be visible to accessors from `<ranges>` header.

namespace nest {
struct StructWithGlobalAccess {};
} // namespace nest

int* begin(nest::StructWithGlobalAccess);
int* end(nest::StructWithGlobalAccess);
int* rbegin(nest::StructWithGlobalAccess);
int* rend(nest::StructWithGlobalAccess);
unsigned int size(nest::StructWithGlobalAccess);

#endif // TEST_STD_RANGES_RANGE_ACCESS_ORDINARY_UNQUALIFIED_LOOKUP_HELPERS_H
