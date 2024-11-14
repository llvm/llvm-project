//===- unittest/Format/FormatTestUtils.h - Formatting unit tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines utility functions for Clang-Format related tests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_FORMAT_FORMATTESTUTILS_H
#define LLVM_CLANG_UNITTESTS_FORMAT_FORMATTESTUTILS_H

#include "clang/Format/Format.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace format {
namespace test {

inline FormatStyle getGoogleStyle() {
  return getGoogleStyle(FormatStyle::LK_Cpp);
}

// When HandleHash is false, preprocessor directives starting with hash will not
// be on separate lines.  This is needed because Verilog uses hash for other
// purposes.
inline std::string messUp(StringRef Code, bool HandleHash = true) {
  std::string MessedUp(Code.str());
  bool InComment = false;
  bool InPreprocessorDirective = false;
  bool JustReplacedNewline = false;
  for (unsigned i = 0, e = MessedUp.size() - 1; i != e; ++i) {
    if (MessedUp[i] == '/' && MessedUp[i + 1] == '/') {
      if (JustReplacedNewline)
        MessedUp[i - 1] = '\n';
      InComment = true;
    } else if (HandleHash && MessedUp[i] == '#' &&
               (JustReplacedNewline || i == 0 || MessedUp[i - 1] == '\n')) {
      if (i != 0)
        MessedUp[i - 1] = '\n';
      InPreprocessorDirective = true;
    } else if (MessedUp[i] == '\\' && MessedUp[i + 1] == '\n') {
      MessedUp[i] = ' ';
      MessedUp[i + 1] = ' ';
    } else if (MessedUp[i] == '\n') {
      if (InComment) {
        InComment = false;
      } else if (InPreprocessorDirective) {
        InPreprocessorDirective = false;
      } else {
        JustReplacedNewline = true;
        MessedUp[i] = ' ';
      }
    } else if (MessedUp[i] != ' ') {
      JustReplacedNewline = false;
    }
  }
  std::string WithoutWhitespace;
  if (MessedUp[0] != ' ')
    WithoutWhitespace.push_back(MessedUp[0]);
  for (unsigned i = 1, e = MessedUp.size(); i != e; ++i)
    if (MessedUp[i] != ' ' || MessedUp[i - 1] != ' ')
      WithoutWhitespace.push_back(MessedUp[i]);
  return WithoutWhitespace;
}

} // end namespace test
} // end namespace format
} // end namespace clang

#endif
