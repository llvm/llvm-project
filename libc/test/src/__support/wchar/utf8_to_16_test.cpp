//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains CharacterConverter unit tests for UTF-8 to UTF-16.
///
//===----------------------------------------------------------------------===//

#include "src/__support/error_or.h"
#include "src/__support/macros/properties/types.h"
#include "src/__support/wchar/character_converter.h"
#include "src/__support/wchar/mbstate.h"
#include "test/UnitTest/Test.h"

#if defined(LIBC_TYPES_WCHAR_T_IS_UTF16)
using TestCharTypesUTF16 = LIBC_NAMESPACE::testing::TypeList<char16_t, wchar_t>;
#else
using TestCharTypesUTF16 = LIBC_NAMESPACE::testing::TypeList<char16_t>;
#endif

TYPED_TEST(LlvmLibcCharacterConverterUTF8To16Test, PopFails,
           TestCharTypesUTF16) {
  using CharType16 = ParamType;

  LIBC_NAMESPACE::internal::mbstate State;
  LIBC_NAMESPACE::internal::CharacterConverter CharConv(&State);
  CharConv.clear();
  ASSERT_EQ(CharConv.push(static_cast<char8_t>('A')), 0);

  ASSERT_FALSE(CharConv.pop<CharType16>().has_value());
}
