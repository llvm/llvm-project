//===-- Serializer.h - ClangDoc Serializer ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the serializing functions fro the clang-doc tool. Given
// a particular declaration, it collects the appropriate information and returns
// a serialized bitcode string for the declaration.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SERIALIZE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SERIALIZE_H

#include "Representation.h"
#include "clang/AST/AST.h"
#include "clang/AST/CommentVisitor.h"
#include <string>
#include <vector>

using namespace clang::comments;

namespace clang {
namespace doc {
namespace serialize {

std::unique_ptr<Info> emitInfo(const NamespaceDecl *D, const FullComment *FC,
                               int LineNumber, StringRef File, bool PublicOnly);
std::unique_ptr<Info> emitInfo(const RecordDecl *D, const FullComment *FC,
                               int LineNumber, StringRef File, bool PublicOnly);
std::unique_ptr<Info> emitInfo(const EnumDecl *D, const FullComment *FC,
                               int LineNumber, StringRef File, bool PublicOnly);
std::unique_ptr<Info> emitInfo(const FunctionDecl *D, const FullComment *FC,
                               int LineNumber, StringRef File, bool PublicOnly);
std::unique_ptr<Info> emitInfo(const CXXMethodDecl *D, const FullComment *FC,
                               int LineNumber, StringRef File, bool PublicOnly);

// Function to hash a given USR value for storage.
// As USRs (Unified Symbol Resolution) could be large, especially for functions
// with long type arguments, we use 160-bits SHA1(USR) values to
// guarantee the uniqueness of symbols while using a relatively small amount of
// memory (vs storing USRs directly).
SymbolID hashUSR(llvm::StringRef USR);

std::string serialize(std::unique_ptr<Info> &I);

} // namespace serialize
} // namespace doc
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SERIALIZE_H
