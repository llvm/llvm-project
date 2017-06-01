//===--- Comments.cpp - Comment manipulation  -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Declares comment manipulation functionality.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_FORMAT_COMMENTS_H
#define LLVM_CLANG_LIB_FORMAT_COMMENTS_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace format {

/// \brief Returns the comment prefix of the line comment \p Comment.
///
/// The comment prefix consists of a leading known prefix, like "//" or "///",
/// together with the following whitespace.
StringRef getLineCommentIndentPrefix(StringRef Comment);

} // namespace format
} // namespace clang

#endif
