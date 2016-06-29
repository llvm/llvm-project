//===- Strings.cpp -------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Strings.h"
#include "Error.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

using namespace llvm;
using namespace lld;
using namespace lld::elf;

// Returns true if S matches T. S can contain glob meta-characters.
// The asterisk ('*') matches zero or more characters, and the question
// mark ('?') matches one character.
bool elf::globMatch(StringRef S, StringRef T) {
  for (;;) {
    if (S.empty())
      return T.empty();
    if (S[0] == '*') {
      S = S.substr(1);
      if (S.empty())
        // Fast path. If a pattern is '*', it matches anything.
        return true;
      for (size_t I = 0, E = T.size(); I < E; ++I)
        if (globMatch(S, T.substr(I)))
          return true;
      return false;
    }
    if (T.empty() || (S[0] != T[0] && S[0] != '?'))
      return false;
    S = S.substr(1);
    T = T.substr(1);
  }
}

// Converts a hex string (e.g. "deadbeef") to a vector.
std::vector<uint8_t> elf::parseHex(StringRef S) {
  std::vector<uint8_t> Hex;
  while (!S.empty()) {
    StringRef B = S.substr(0, 2);
    S = S.substr(2);
    uint8_t H;
    if (B.getAsInteger(16, H)) {
      error("not a hexadecimal value: " + B);
      return {};
    }
    Hex.push_back(H);
  }
  return Hex;
}
