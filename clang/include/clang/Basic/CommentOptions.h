//===- CommentOptions.h - Options for parsing comments ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines the clang::CommentOptions interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_COMMENTOPTIONS_H
#define LLVM_CLANG_BASIC_COMMENTOPTIONS_H

#include <string>
#include <vector>

namespace clang {

/// Options for controlling comment parsing.
struct CommentOptions {
  using BlockCommandNamesTy = std::vector<std::string>;

#define TYPED_COMMENTOPT(Type, Name, Description) Type Name;
#include "clang/Basic/CommentOptions.def"

  CommentOptions() : ParseAllComments(false) {}
};

} // namespace clang

#endif // LLVM_CLANG_BASIC_COMMENTOPTIONS_H
