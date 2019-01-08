//===--- SymbolID.cpp --------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolID.h"
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
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Bad ID length");
  for (char C : Str)
    if (!llvm::isHexDigit(C))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Bad hex ID");
  return fromRaw(llvm::fromHex(Str));
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const SymbolID &ID) {
  return OS << llvm::toHex(ID.raw());
}

llvm::hash_code hash_value(const SymbolID &ID) {
  // We already have a good hash, just return the first bytes.
  assert(sizeof(size_t) <= SymbolID::RawSize && "size_t longer than SHA1!");
  size_t Result;
  memcpy(&Result, ID.raw().data(), sizeof(size_t));
  return llvm::hash_code(Result);
}

} // namespace clangd
} // namespace clang
