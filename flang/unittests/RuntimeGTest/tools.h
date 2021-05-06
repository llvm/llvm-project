//===-- flang/unittests/RuntimeGTest/tools.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_UNITTESTS_RUNTIME_TOOLS_H_
#define FORTRAN_UNITTESTS_RUNTIME_TOOLS_H_

#include "gtest/gtest.h"
#include "../../runtime/allocatable.h"
#include "../../runtime/cpp-type.h"
#include "../../runtime/descriptor.h"
#include "../../runtime/type-code.h"
#include <cstdint>
#include <cstring>
#include <vector>

namespace Fortran::runtime {

template <typename A>
static void StoreElement(void *p, const A &x, std::size_t bytes) {
  std::memcpy(p, &x, bytes);
}

template <typename CHAR>
static void StoreElement(
    void *p, const std::basic_string<CHAR> &str, std::size_t bytes) {
  ASSERT_LE(bytes, sizeof(CHAR) * str.size());
  std::memcpy(p, str.data(), bytes);
}

template <TypeCategory CAT, int KIND, typename A>
static OwningPtr<Descriptor> MakeArray(const std::vector<int> &shape,
    const std::vector<A> &data,
    std::size_t elemLen = CAT == TypeCategory::Complex ? 2 * KIND : KIND) {
  auto rank{static_cast<int>(shape.size())};
  auto result{Descriptor::Create(TypeCode{CAT, KIND}, elemLen, nullptr, rank,
      nullptr, CFI_attribute_allocatable)};
  for (int j{0}; j < rank; ++j) {
    result->GetDimension(j).SetBounds(1, shape[j]);
  }
  int stat{result->Allocate()};
  EXPECT_EQ(stat, 0) << stat;
  EXPECT_LE(data.size(), result->Elements());
  char *p{result->OffsetElement<char>()};
  for (A x : data) {
    StoreElement(p, x, elemLen);
    p += elemLen;
  }
  return result;
}

} // namespace Fortran::runtime
#endif // FORTRAN_UNITTESTS_RUNTIME_TOOLS_H_
