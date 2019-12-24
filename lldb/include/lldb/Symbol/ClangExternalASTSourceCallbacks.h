//===-- ClangExternalASTSourceCallbacks.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangExternalASTSourceCallbacks_h_
#define liblldb_ClangExternalASTSourceCallbacks_h_

#include "lldb/Symbol/ClangASTContext.h"
#include "clang/AST/ExternalASTSource.h"

namespace lldb_private {

class ClangASTContext;

class ClangExternalASTSourceCallbacks : public clang::ExternalASTSource {
public:
  ClangExternalASTSourceCallbacks(ClangASTContext &ast) : m_ast(ast) {}

  void FindExternalLexicalDecls(
      const clang::DeclContext *DC,
      llvm::function_ref<bool(clang::Decl::Kind)> IsKindWeWant,
      llvm::SmallVectorImpl<clang::Decl *> &Result) override;

  void CompleteType(clang::TagDecl *tag_decl) override;

  void CompleteType(clang::ObjCInterfaceDecl *objc_decl) override;

  bool layoutRecordType(
      const clang::RecordDecl *Record, uint64_t &Size, uint64_t &Alignment,
      llvm::DenseMap<const clang::FieldDecl *, uint64_t> &FieldOffsets,
      llvm::DenseMap<const clang::CXXRecordDecl *, clang::CharUnits>
          &BaseOffsets,
      llvm::DenseMap<const clang::CXXRecordDecl *, clang::CharUnits>
          &VirtualBaseOffsets) override;

private:
  ClangASTContext &m_ast;
};

} // namespace lldb_private

#endif // liblldb_ClangExternalASTSourceCallbacks_h_
