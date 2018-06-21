//===--- CommentOptions.h - Options for parsing comments -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the clang::CommentOptions interface.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_COMMENTOPTIONS_H
#define LLVM_CLANG_BASIC_COMMENTOPTIONS_H

#include <string>
#include <vector>

namespace clang {

/// \brief Options for controlling comment parsing.
struct CommentOptions {
  typedef std::vector<std::string> BlockCommandNamesTy;

  /// \brief Command names to treat as block commands in comments.
  /// Should not include the leading backslash.
  BlockCommandNamesTy BlockCommandNames;

  /// \brief Treat ordinary comments as documentation comments.
  bool ParseAllComments;

  CommentOptions() : ParseAllComments(false) { }
};

}  // end namespace clang

#endif
