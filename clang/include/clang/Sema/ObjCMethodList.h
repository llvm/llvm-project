//===--- ObjCMethodList.h - A singly linked list of methods -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines ObjCMethodList, a singly-linked list of methods.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_OBJC_METHOD_LIST_H
#define LLVM_CLANG_SEMA_OBJC_METHOD_LIST_H

namespace clang {

class ObjCMethodDecl;

/// ObjCMethodList - a linked list of methods with different signatures.
struct ObjCMethodList {
  ObjCMethodDecl *Method;
  ObjCMethodList *Next;

  ObjCMethodList() {
    Method = 0;
    Next = 0;
  }
  ObjCMethodList(ObjCMethodDecl *M, ObjCMethodList *C) {
    Method = M;
    Next = C;
  }
};

}

#endif
