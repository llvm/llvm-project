//===--- Comments.cpp - Comment Manipulation  -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Implements comment manipulation.
///
//===----------------------------------------------------------------------===//

#include "Comments.h"

namespace clang {
namespace format {

StringRef getLineCommentIndentPrefix(StringRef Comment) {
  static const char *const KnownPrefixes[] = {"///", "//", "//!"};
  StringRef LongestPrefix;
  for (StringRef KnownPrefix : KnownPrefixes) {
    if (Comment.startswith(KnownPrefix)) {
      size_t PrefixLength = KnownPrefix.size();
      while (PrefixLength < Comment.size() && Comment[PrefixLength] == ' ')
        ++PrefixLength;
      if (PrefixLength > LongestPrefix.size())
        LongestPrefix = Comment.substr(0, PrefixLength);
    }
  }
  return LongestPrefix;
}

} // namespace format
} // namespace clang
