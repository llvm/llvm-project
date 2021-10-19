//===--- DumpAST.h - Serialize clang AST to LSP -----------------*- C++-*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exposing clang's AST can give insight into the precise meaning of code.
// (C++ is a complicated language, and very few people know all its rules).
// Despite the name, clang's AST describes *semantics* and so includes nodes
// for implicit behavior like conversions.
//
// It's also useful to developers who work with the clang AST specifically,
// and want to know how certain constructs are represented.
//
// The main representation is not based on the familiar -ast-dump output,
// which is heavy on internal details.
// It also does not use the -ast-dump=json output, which captures the same
// detail in a machine-friendly way, but requires client-side logic to present.
// Instead, the key information is bundled into a few fields (role/kind/detail)
// with weakly-defined semantics, optimized for easy presentation.
// The -ast-dump output is preserved in the 'arcana' field, and may be shown
// e.g. as a tooltip.
//
// The textDocument/ast method implemented here is a clangd extension.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_DUMPAST_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_DUMPAST_H

#include "Protocol.h"
#include "clang/AST/ASTContext.h"

namespace clang {
namespace syntax {
class TokenBuffer;
} // namespace syntax
namespace clangd {

// Note: It's safe for the node to be a TranslationUnitDecl, as this function
//       does not deserialize the preamble.
ASTNode dumpAST(const DynTypedNode &, const syntax::TokenBuffer &Tokens,
                const ASTContext &);

} // namespace clangd
} // namespace clang

#endif
