//===-- PragmaCommentHandler.h - find all symbols----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_PRAGMA_COMMENT_HANDLER_H
#define LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_PRAGMA_COMMENT_HANDLER_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Preprocessor.h"
#include <map>

namespace clang {
namespace find_all_symbols {

class HeaderMapCollector;

/// PragmaCommentHandler parses pragma comment on include files to
/// determine when we should include a different header from the header that
/// directly defines a symbol.
///
/// Currently it only supports IWYU private pragma:
/// https://github.com/include-what-you-use/include-what-you-use/blob/master/docs/IWYUPragmas.md#iwyu-pragma-private
class PragmaCommentHandler : public clang::CommentHandler {
public:
  PragmaCommentHandler(HeaderMapCollector *Collector) : Collector(Collector) {}

  bool HandleComment(Preprocessor &PP, SourceRange Range) override;

private:
  HeaderMapCollector *const Collector;
};

} // namespace find_all_symbols
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_PRAGMA_COMMENT_HANDLER_H
