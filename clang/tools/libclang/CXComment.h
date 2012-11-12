//===- CXComment.h - Routines for manipulating CXComments -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines routines for manipulating CXComments.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CXCOMMENT_H
#define LLVM_CLANG_CXCOMMENT_H

#include "clang-c/Index.h"
#include "CXTranslationUnit.h"

#include "clang/AST/Comment.h"
#include "clang/AST/ASTContext.h"
#include "clang/Frontend/ASTUnit.h"

namespace clang {
namespace comments {
  class CommandTraits;
}

namespace cxcomment {

inline CXComment createCXComment(const comments::Comment *C,
                                 CXTranslationUnit TU) {
  CXComment Result;
  Result.ASTNode = C;
  Result.TranslationUnit = TU;
  return Result;
}

inline const comments::Comment *getASTNode(CXComment CXC) {
  return static_cast<const comments::Comment *>(CXC.ASTNode);
}

template<typename T>
inline const T *getASTNodeAs(CXComment CXC) {
  const comments::Comment *C = getASTNode(CXC);
  if (!C)
    return NULL;

  return dyn_cast<T>(C);
}

inline ASTContext &getASTContext(CXComment CXC) {
  return static_cast<ASTUnit *>(CXC.TranslationUnit->TUData)->getASTContext();
}

inline comments::CommandTraits &getCommandTraits(CXComment CXC) {
  return getASTContext(CXC).getCommentCommandTraits();
}

} // end namespace cxcomment
} // end namespace clang

#endif

