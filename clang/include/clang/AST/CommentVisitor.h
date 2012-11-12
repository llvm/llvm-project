//===--- CommentVisitor.h - Visitor for Comment subclasses ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Comment.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang {
namespace comments {

template <typename T> struct make_ptr       { typedef       T *type; };
template <typename T> struct make_const_ptr { typedef const T *type; };

template<template <typename> class Ptr, typename ImplClass, typename RetTy=void>
class CommentVisitorBase {
public:
#define PTR(CLASS) typename Ptr<CLASS>::type
#define DISPATCH(NAME, CLASS) \
 return static_cast<ImplClass*>(this)->visit ## NAME(static_cast<PTR(CLASS)>(C))

  RetTy visit(PTR(Comment) C) {
    if (!C)
      return RetTy();

    switch (C->getCommentKind()) {
    default: llvm_unreachable("Unknown comment kind!");
#define ABSTRACT_COMMENT(COMMENT)
#define COMMENT(CLASS, PARENT) \
    case Comment::CLASS##Kind: DISPATCH(CLASS, CLASS);
#include "clang/AST/CommentNodes.inc"
#undef ABSTRACT_COMMENT
#undef COMMENT
    }
  }

  // If the derived class does not implement a certain Visit* method, fall back
  // on Visit* method for the superclass.
#define ABSTRACT_COMMENT(COMMENT) COMMENT
#define COMMENT(CLASS, PARENT) \
  RetTy visit ## CLASS(PTR(CLASS) C) { DISPATCH(PARENT, PARENT); }
#include "clang/AST/CommentNodes.inc"
#undef ABSTRACT_COMMENT
#undef COMMENT

  RetTy visitComment(PTR(Comment) C) { return RetTy(); }

#undef PTR
#undef DISPATCH
};

template<typename ImplClass, typename RetTy=void>
class CommentVisitor :
    public CommentVisitorBase<make_ptr, ImplClass, RetTy> {};

template<typename ImplClass, typename RetTy=void>
class ConstCommentVisitor :
    public CommentVisitorBase<make_const_ptr, ImplClass, RetTy> {};

} // end namespace comments
} // end namespace clang

