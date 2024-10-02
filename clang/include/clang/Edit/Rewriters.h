//===--- Rewriters.h - Rewritings     ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_EDIT_REWRITERS_H
#define LLVM_CLANG_EDIT_REWRITERS_H

#include "clang/Support/Compiler.h"
namespace clang {
  class ObjCMessageExpr;
  class ObjCMethodDecl;
  class ObjCInterfaceDecl;
  class ObjCProtocolDecl;
  class NSAPI;
  class EnumDecl;
  class TypedefDecl;
  class ParentMap;

namespace edit {
  class Commit;

CLANG_ABI bool rewriteObjCRedundantCallWithLiteral(const ObjCMessageExpr *Msg,
                                         const NSAPI &NS, Commit &commit);

CLANG_ABI bool rewriteToObjCLiteralSyntax(const ObjCMessageExpr *Msg,
                                const NSAPI &NS, Commit &commit,
                                const ParentMap *PMap);

CLANG_ABI bool rewriteToObjCSubscriptSyntax(const ObjCMessageExpr *Msg,
                                  const NSAPI &NS, Commit &commit);

}

}  // end namespace clang

#endif
