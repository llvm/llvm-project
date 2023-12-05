//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_LANGUAGE_SUPPORT_CMP_CMP_ALG_ORDINARY_UNQUALIFIED_LOOKUP_HELPERS_H
#define TEST_STD_LANGUAGE_SUPPORT_CMP_CMP_ALG_ORDINARY_UNQUALIFIED_LOOKUP_HELPERS_H

// Note: this header should be included before any other header.
// Comparison functions defined here must be visible to CPOs from `<compare>` header.

namespace nest {
struct StructWithGlobalCmpFunctions {};
} // namespace nest

struct ConvertibleToComparisonType;

ConvertibleToComparisonType strong_order(nest::StructWithGlobalCmpFunctions, nest::StructWithGlobalCmpFunctions);
ConvertibleToComparisonType weak_order(nest::StructWithGlobalCmpFunctions, nest::StructWithGlobalCmpFunctions);
ConvertibleToComparisonType partial_order(nest::StructWithGlobalCmpFunctions, nest::StructWithGlobalCmpFunctions);

#include <compare> // Intentionally included here, so we can define `ConvertibleToComparisonType` later.

struct ConvertibleToComparisonType {
  operator std::strong_ordering() const;
  operator std::weak_ordering() const;
  operator std::partial_ordering() const;
};

#endif // TEST_STD_LANGUAGE_SUPPORT_CMP_CMP_ALG_ORDINARY_UNQUALIFIED_LOOKUP_HELPERS_H
