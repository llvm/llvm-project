//===-- Serializer.h - ClangDoc Serializer ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

std::string emitInfo(const NamespaceDecl *D, const FullComment *FC,
                     int LineNumber, StringRef File);
std::string emitInfo(const RecordDecl *D, const FullComment *FC, int LineNumber,
                     StringRef File);
std::string emitInfo(const EnumDecl *D, const FullComment *FC, int LineNumber,
                     StringRef File);
std::string emitInfo(const FunctionDecl *D, const FullComment *FC,
                     int LineNumber, StringRef File);
std::string emitInfo(const CXXMethodDecl *D, const FullComment *FC,
                     int LineNumber, StringRef File);

// Function to hash a given USR value for storage.
// As USRs (Unified Symbol Resolution) could be large, especially for functions
// with long type arguments, we use 160-bits SHA1(USR) values to
// guarantee the uniqueness of symbols while using a relatively small amount of
// memory (vs storing USRs directly).
SymbolID hashUSR(llvm::StringRef USR);

} // namespace serialize
} // namespace doc
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SERIALIZE_H
