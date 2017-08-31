//===- unittest/Format/FormatTestUtils.h - Formatting unit tests ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines utility functions for Clang-Format related tests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_FORMAT_FORMATTESTUTILS_H
#define LLVM_CLANG_UNITTESTS_FORMAT_FORMATTESTUTILS_H

#include "llvm/ADT/StringRef.h"

namespace clang {
namespace format {
namespace test {

inline std::string messUp(llvm::StringRef Code) {
  std::string MessedUp(Code.str());
  bool InComment = false;
  bool InPreprocessorDirective = false;
  bool JustReplacedNewline = false;
  for (unsigned i = 0, e = MessedUp.size() - 1; i != e; ++i) {
    if (MessedUp[i] == '/' && MessedUp[i + 1] == '/') {
      if (JustReplacedNewline)
        MessedUp[i - 1] = '\n';
      InComment = true;
    } else if (MessedUp[i] == '#' &&
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
  for (unsigned i = 1, e = MessedUp.size(); i != e; ++i) {
    if (MessedUp[i] != ' ' || MessedUp[i - 1] != ' ')
      WithoutWhitespace.push_back(MessedUp[i]);
  }
  return WithoutWhitespace;
}

} // end namespace test
} // end namespace format
} // end namespace clang

#endif
