//===- Strings.h ------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_STRINGS_H
#define LLD_STRINGS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/GlobPattern.h"
#include <string>
#include <vector>

namespace lld {
// Returns a demangled C++ symbol name. If Name is not a mangled
// name, it returns Optional::None.
llvm::Optional<std::string> demangleItanium(llvm::StringRef Name);
llvm::Optional<std::string> demangleMSVC(llvm::StringRef S);

std::vector<uint8_t> parseHex(llvm::StringRef S);
bool isValidCIdentifier(llvm::StringRef S);

// This is a lazy version of StringRef. String size is computed lazily
// when it is needed. It is more efficient than StringRef to instantiate
// if you have a string whose size is unknown.
//
// COFF and ELF string tables contain a lot of null-terminated strings.
// Most of them are not necessary for the linker because they are names
// of local symbols and the linker doesn't use local symbol names for
// name resolution. So, we use this class to represents strings read
// from string tables.
class StringRefZ {
public:
  StringRefZ() : Start(nullptr), Size(0) {}
  StringRefZ(const char *S, size_t Size) : Start(S), Size(Size) {}

  /*implicit*/ StringRefZ(const char *S) : Start(S), Size(-1) {}

  /*implicit*/ StringRefZ(llvm::StringRef S)
      : Start(S.data()), Size(S.size()) {}

  operator llvm::StringRef() const {
    if (Size == (size_t)-1)
      Size = strlen(Start);
    return {Start, Size};
  }

private:
  const char *Start;
  mutable size_t Size;
};

// This class represents multiple glob patterns.
class StringMatcher {
public:
  StringMatcher() = default;
  explicit StringMatcher(llvm::ArrayRef<llvm::StringRef> Pat);

  bool match(llvm::StringRef S) const;

private:
  std::vector<llvm::GlobPattern> Patterns;
};

inline llvm::ArrayRef<uint8_t> toArrayRef(llvm::StringRef S) {
  return {reinterpret_cast<const uint8_t *>(S.data()), S.size()};
}
}

#endif
