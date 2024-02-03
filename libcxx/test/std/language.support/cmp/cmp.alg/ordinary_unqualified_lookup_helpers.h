//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_LANGUAGE_SUPPORT_CMP_CMP_ALG_ORDINARY_UNQUALIFIED_LOOKUP_HELPERS_H
#define TEST_STD_LANGUAGE_SUPPORT_CMP_CMP_ALG_ORDINARY_UNQUALIFIED_LOOKUP_HELPERS_H

namespace ordinary_unqualified_lookup_helpers {
struct StructWithGlobalCmpFunctions {};
} // namespace ordinary_unqualified_lookup_helpers

struct ConvertibleToCmpType;

ConvertibleToCmpType strong_order(ordinary_unqualified_lookup_helpers::StructWithGlobalCmpFunctions,
                                  ordinary_unqualified_lookup_helpers::StructWithGlobalCmpFunctions);
ConvertibleToCmpType weak_order(ordinary_unqualified_lookup_helpers::StructWithGlobalCmpFunctions,
                                ordinary_unqualified_lookup_helpers::StructWithGlobalCmpFunctions);
ConvertibleToCmpType partial_order(ordinary_unqualified_lookup_helpers::StructWithGlobalCmpFunctions,
                                   ordinary_unqualified_lookup_helpers::StructWithGlobalCmpFunctions);

#include <compare> // Intentionally included here, so we can define `ConvertibleToCmpType` later.

struct ConvertibleToCmpType {
  operator std::strong_ordering() const;
  operator std::weak_ordering() const;
  operator std::partial_ordering() const;
};

#endif // TEST_STD_LANGUAGE_SUPPORT_CMP_CMP_ALG_ORDINARY_UNQUALIFIED_LOOKUP_HELPERS_H
