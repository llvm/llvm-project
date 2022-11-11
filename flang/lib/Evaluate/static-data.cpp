//===-- lib/Evaluate/static-data.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/static-data.h"
#include "flang/Parser/characters.h"

namespace Fortran::evaluate {

llvm::raw_ostream &StaticDataObject::AsFortran(
    llvm::raw_ostream &o, bool bigEndian) const {
  if (auto string{AsString()}) {
    o << parser::QuoteCharacterLiteral(*string);
  } else if (auto string{AsU16String(bigEndian)}) {
    o << "2_" << parser::QuoteCharacterLiteral(*string);
  } else if (auto string{AsU32String(bigEndian)}) {
    o << "4_" << parser::QuoteCharacterLiteral(*string);
  } else {
    CRASH_NO_CASE;
  }
  return o;
}

StaticDataObject &StaticDataObject::Push(const std::string &string, bool) {
  for (auto ch : string) {
    data_.push_back(static_cast<std::uint8_t>(ch));
  }
  return *this;
}

StaticDataObject &StaticDataObject::Push(
    const std::u16string &string, bool bigEndian) {
  int shift{bigEndian ? 8 : 0};
  for (auto ch : string) {
    data_.push_back(static_cast<std::uint8_t>(ch >> shift));
    data_.push_back(static_cast<std::uint8_t>(ch >> (shift ^ 8)));
  }
  return *this;
}

StaticDataObject &StaticDataObject::Push(
    const std::u32string &string, bool bigEndian) {
  int shift{bigEndian ? 24 : 0};
  for (auto ch : string) {
    data_.push_back(static_cast<std::uint8_t>(ch >> shift));
    data_.push_back(static_cast<std::uint8_t>(ch >> (shift ^ 8)));
    data_.push_back(static_cast<std::uint8_t>(ch >> (shift ^ 16)));
    data_.push_back(static_cast<std::uint8_t>(ch >> (shift ^ 24)));
  }
  return *this;
}

std::optional<std::string> StaticDataObject::AsString() const {
  if (itemBytes_ <= 1) {
    std::string result;
    for (std::uint8_t byte : data_) {
      result += static_cast<char>(byte);
    }
    return {std::move(result)};
  }
  return std::nullopt;
}

std::optional<std::u16string> StaticDataObject::AsU16String(
    bool bigEndian) const {
  if (itemBytes_ == 2) {
    int shift{bigEndian ? 8 : 0};
    std::u16string result;
    auto end{data_.cend()};
    for (auto byte{data_.cbegin()}; byte < end;) {
      result += static_cast<char16_t>(*byte++) << shift |
          static_cast<char16_t>(*byte++) << (shift ^ 8);
    }
    return {std::move(result)};
  }
  return std::nullopt;
}

std::optional<std::u32string> StaticDataObject::AsU32String(
    bool bigEndian) const {
  if (itemBytes_ == 4) {
    int shift{bigEndian ? 24 : 0};
    std::u32string result;
    auto end{data_.cend()};
    for (auto byte{data_.cbegin()}; byte < end;) {
      result += static_cast<char32_t>(*byte++) << shift |
          static_cast<char32_t>(*byte++) << (shift ^ 8) |
          static_cast<char32_t>(*byte++) << (shift ^ 16) |
          static_cast<char32_t>(*byte++) << (shift ^ 24);
    }
    return {std::move(result)};
  }
  return std::nullopt;
}
} // namespace Fortran::evaluate
