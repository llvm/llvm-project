//===--- MacroArgs.h - Formal argument info for Macros ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MacroArgs interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_PPEMBEDPARAMETERS_H
#define LLVM_CLANG_LEX_PPEMBEDPARAMETERS_H

#include "clang/Lex/PPDirectiveParameter.h"
#include "clang/Lex/Token.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {

/// Preprocessor extension embed parameter "clang::offset"
/// `clang::offset( constant-expression )`
class PPEmbedParameterOffset : public PPDirectiveParameter {
public:
  size_t Offset;

  PPEmbedParameterOffset(size_t Offset, SourceLocation Start,
                         SourceLocation End)
      : Offset(Offset), PPDirectiveParameter(Start, End) {}
};

/// Preprocessor standard embed parameter "limit"
/// `limit( constant-expression )`
class PPEmbedParameterLimit : public PPDirectiveParameter {
public:
  size_t Limit;

  PPEmbedParameterLimit(size_t Limit, SourceLocation Start, SourceLocation End)
      : Limit(Limit), PPDirectiveParameter(Start, End) {}
};

/// Preprocessor standard embed parameter "prefix"
/// `prefix( balanced-token-seq )`
class PPEmbedParameterPrefix : public PPDirectiveParameter {
public:
  SmallVector<Token, 2> Tokens;

  PPEmbedParameterPrefix(SmallVector<Token, 2> Tokens, SourceLocation Start,
                         SourceLocation End)
      : Tokens(std::move(Tokens)), PPDirectiveParameter(Start, End) {}
};

/// Preprocessor standard embed parameter "suffix"
/// `suffix( balanced-token-seq )`
class PPEmbedParameterSuffix : public PPDirectiveParameter {
public:
  SmallVector<Token, 2> Tokens;

  PPEmbedParameterSuffix(SmallVector<Token, 2> Tokens, SourceLocation Start,
                         SourceLocation End)
      : Tokens(std::move(Tokens)), PPDirectiveParameter(Start, End) {}
};

/// Preprocessor standard embed parameter "if_empty"
/// `if_empty( balanced-token-seq )`
class PPEmbedParameterIfEmpty : public PPDirectiveParameter {
public:
  SmallVector<Token, 2> Tokens;

  PPEmbedParameterIfEmpty(SmallVector<Token, 2> Tokens, SourceLocation Start,
                          SourceLocation End)
      : Tokens(std::move(Tokens)), PPDirectiveParameter(Start, End) {}
};

} // end namespace clang

#endif
