//===--- SymbolID.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolID.h"
#include "support/Logger.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA1.h"

namespace clang {
namespace clangd {

SymbolID::SymbolID(llvm::StringRef USR) {
  auto Hash = llvm::SHA1::hash(llvm::arrayRefFromStringRef(USR));
  static_assert(sizeof(Hash) >= RawSize, "RawSize larger than SHA1");
  memcpy(HashValue.data(), Hash.data(), RawSize);
}

llvm::StringRef SymbolID::raw() const {
  return llvm::StringRef(reinterpret_cast<const char *>(HashValue.data()),
                         RawSize);
}

SymbolID SymbolID::fromRaw(llvm::StringRef Raw) {
  SymbolID ID;
  assert(Raw.size() == RawSize);
  memcpy(ID.HashValue.data(), Raw.data(), RawSize);
  return ID;
}

std::string SymbolID::str() const { return llvm::toHex(raw()); }

llvm::Expected<SymbolID> SymbolID::fromStr(llvm::StringRef Str) {
  if (Str.size() != RawSize * 2)
    return error("Bad ID length");
  for (char C : Str)
    if (!llvm::isHexDigit(C))
      return error("Bad hex ID");
  return fromRaw(llvm::fromHex(Str));
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const SymbolID &ID) {
  return OS << llvm::toHex(ID.raw());
}

} // namespace clangd
} // namespace clang
