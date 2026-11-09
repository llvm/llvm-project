//===--- PPEmbedParameters.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines all of the preprocessor directive parameters for #embed
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_PPEMBEDPARAMETERS_H
#define LLVM_CLANG_LEX_PPEMBEDPARAMETERS_H

#include "clang/Lex/Token.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {

/// Preprocessor extension embed parameter "clang::offset"
/// `clang::offset( constant-expression )`
class PPEmbedParameterOffset {
public:
  size_t Offset;

  explicit PPEmbedParameterOffset(size_t Offset) : Offset(Offset) {}
};

/// Preprocessor standard embed parameter "limit"
/// `limit( constant-expression )`
class PPEmbedParameterLimit {
public:
  size_t Limit;

  explicit PPEmbedParameterLimit(size_t Limit) : Limit(Limit) {}
};

/// Preprocessor standard embed parameter "prefix"
/// `prefix( balanced-token-seq )`
class PPEmbedParameterPrefix {
public:
  SmallVector<Token, 2> Tokens;

  explicit PPEmbedParameterPrefix(SmallVectorImpl<Token> &&Tokens)
      : Tokens(std::move(Tokens)) {}
};

/// Preprocessor standard embed parameter "suffix"
/// `suffix( balanced-token-seq )`
class PPEmbedParameterSuffix {
public:
  SmallVector<Token, 2> Tokens;

  explicit PPEmbedParameterSuffix(SmallVectorImpl<Token> &&Tokens)
      : Tokens(std::move(Tokens)) {}
};

/// Preprocessor standard embed parameter "if_empty"
/// `if_empty( balanced-token-seq )`
class PPEmbedParameterIfEmpty {
public:
  SmallVector<Token, 2> Tokens;

  explicit PPEmbedParameterIfEmpty(SmallVectorImpl<Token> &&Tokens)
      : Tokens(std::move(Tokens)) {}
};

struct LexEmbedParametersResult {
  std::optional<PPEmbedParameterLimit> MaybeLimitParam;
  std::optional<PPEmbedParameterOffset> MaybeOffsetParam;
  std::optional<PPEmbedParameterIfEmpty> MaybeIfEmptyParam;
  std::optional<PPEmbedParameterPrefix> MaybePrefixParam;
  std::optional<PPEmbedParameterSuffix> MaybeSuffixParam;
  int UnrecognizedParams;

  size_t PrefixTokenCount() const {
    if (MaybePrefixParam)
      return MaybePrefixParam->Tokens.size();
    return 0;
  }
  size_t SuffixTokenCount() const {
    if (MaybeSuffixParam)
      return MaybeSuffixParam->Tokens.size();
    return 0;
  }
};
} // end namespace clang

#endif
