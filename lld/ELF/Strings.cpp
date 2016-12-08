//===- Strings.cpp -------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Strings.h"
#include "Config.h"
#include "Error.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/config.h"
#include "llvm/Demangle/Demangle.h"
#include <algorithm>
#include <cstring>

using namespace llvm;
using namespace lld;
using namespace lld::elf;

// This is a scanner for the glob pattern.
// A glob pattern token is one of "*", "?", "[<chars>]", "[^<chars>]"
// (which is a negative form of "[<chars>]"), or a non-meta character.
// This function returns the first token in S.
BitVector GlobPattern::scan(StringRef &S) {
  switch (S[0]) {
  case '*':
    S = S.substr(1);
    // '*' is represented by an empty bitvector.
    // All other bitvectors are 256-bit long.
    return BitVector();
  case '?':
    S = S.substr(1);
    return BitVector(256, true);
  case '[': {
    size_t End = S.find(']', 1);
    if (End == StringRef::npos) {
      error("invalid glob pattern: " + Original);
      return BitVector(256, false);
    }
    StringRef Chars = S.substr(1, End - 1);
    S = S.substr(End + 1);
    if (Chars.startswith("^"))
      return expand(Chars.substr(1)).flip();
    return expand(Chars);
  }
  default:
    BitVector BV(256, false);
    BV[S[0]] = true;
    S = S.substr(1);
    return BV;
  }
}

// Expands character ranges and returns a bitmap.
// For example, "a-cf-hz" is expanded to "abcfghz".
BitVector GlobPattern::expand(StringRef S) {
  BitVector BV(256, false);

  // Expand "x-y".
  for (;;) {
    if (S.size() < 3)
      break;

    // If it doesn't start with something like "x-y",
    // consume the first character and proceed.
    if (S[1] != '-') {
      BV[S[0]] = true;
      S = S.substr(1);
      continue;
    }

    // It must be in the form of "x-y".
    // Validate it and then interpret the range.
    if (S[0] > S[2]) {
      error("invalid glob pattern: " + Original);
      return BV;
    }
    for (int C = S[0]; C <= S[2]; ++C)
      BV[C] = true;
    S = S.substr(3);
  }

  for (char C : S)
    BV[C] = true;
  return BV;
}

GlobPattern::GlobPattern(StringRef S) : Original(S) {
  if (!hasWildcard(S)) {
    // S doesn't contain any metacharacter,
    // so the regular string comparison should work.
    Exact = S;
  } else if (S.endswith("*") && !hasWildcard(S.drop_back())) {
    // S is something like "foo*". We can use startswith().
    Prefix = S.drop_back();
  } else if (S.startswith("*") && !hasWildcard(S.drop_front())) {
    // S is something like "*foo". We can use endswith().
    Suffix = S.drop_front();
  } else {
    // Otherwise, we need to do real glob pattern matching.
    // Parse the pattern now.
    while (!S.empty())
      Tokens.push_back(scan(S));
  }
}

bool GlobPattern::match(StringRef S) const {
  if (Exact)
    return S == *Exact;
  if (Prefix)
    return S.startswith(*Prefix);
  if (Suffix)
    return S.endswith(*Suffix);
  return matchOne(Tokens, S);
}

// Runs glob pattern Pats against string S.
bool GlobPattern::matchOne(ArrayRef<BitVector> Pats, StringRef S) const {
  for (;;) {
    if (Pats.empty())
      return S.empty();

    // If Pats[0] is '*', try to match Pats[1..] against all possible
    // substrings of S to see at least one pattern succeeds.
    if (Pats[0].size() == 0) {
      Pats = Pats.slice(1);
      if (Pats.empty())
        // Fast path. If a pattern is '*', it matches anything.
        return true;
      for (size_t I = 0, E = S.size(); I < E; ++I)
        if (matchOne(Pats, S.substr(I)))
          return true;
      return false;
    }

    // If Pats[0] is not '*', it must consume one character.
    if (S.empty() || !Pats[0][S[0]])
      return false;
    Pats = Pats.slice(1);
    S = S.substr(1);
  }
}

StringMatcher::StringMatcher(const std::vector<StringRef> &Pat) {
  for (StringRef S : Pat)
    Patterns.push_back(GlobPattern(S));
}

bool StringMatcher::match(StringRef S) const {
  for (const GlobPattern &Pat : Patterns)
    if (Pat.match(S))
      return true;
  return false;
}

// If an input string is in the form of "foo.N" where N is a number,
// return N. Otherwise, returns 65536, which is one greater than the
// lowest priority.
int elf::getPriority(StringRef S) {
  size_t Pos = S.rfind('.');
  if (Pos == StringRef::npos)
    return 65536;
  int V;
  if (S.substr(Pos + 1).getAsInteger(10, V))
    return 65536;
  return V;
}

bool elf::hasWildcard(StringRef S) {
  return S.find_first_of("?*[") != StringRef::npos;
}

StringRef elf::unquote(StringRef S) {
  if (!S.startswith("\""))
    return S;
  return S.substr(1, S.size() - 2);
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

static bool isAlpha(char C) {
  return ('a' <= C && C <= 'z') || ('A' <= C && C <= 'Z') || C == '_';
}

static bool isAlnum(char C) { return isAlpha(C) || ('0' <= C && C <= '9'); }

// Returns true if S is valid as a C language identifier.
bool elf::isValidCIdentifier(StringRef S) {
  return !S.empty() && isAlpha(S[0]) &&
         std::all_of(S.begin() + 1, S.end(), isAlnum);
}

// Returns the demangled C++ symbol name for Name.
Optional<std::string> elf::demangle(StringRef Name) {
  // __cxa_demangle can be used to demangle strings other than symbol
  // names which do not necessarily start with "_Z". Name can be
  // either a C or C++ symbol. Don't call __cxa_demangle if the name
  // does not look like a C++ symbol name to avoid getting unexpected
  // result for a C symbol that happens to match a mangled type name.
  if (!Name.startswith("_Z"))
    return None;

  char *Buf = itaniumDemangle(Name.str().c_str(), nullptr, nullptr, nullptr);
  if (!Buf)
    return None;
  std::string S(Buf);
  free(Buf);
  return S;
}
