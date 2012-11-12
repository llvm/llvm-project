//= CFGRecStmtDeclVisitor - Recursive visitor of CFG stmts/decls -*- C++ --*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the template class CFGRecStmtDeclVisitor, which extends
// CFGRecStmtVisitor by implementing (typed) visitation of decls.
//
// FIXME: This may not be fully complete.  We currently explore only subtypes
//        of ScopedDecl.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_CFG_REC_STMT_DECL_VISITOR_H
#define LLVM_CLANG_ANALYSIS_CFG_REC_STMT_DECL_VISITOR_H

#include "clang/Analysis/Visitors/CFGRecStmtVisitor.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclCXX.h"

#define DISPATCH_CASE(CLASS)                                  \
case Decl::CLASS:                                             \
static_cast<ImplClass*>(this)->Visit##CLASS##Decl(            \
                               static_cast<CLASS##Decl*>(D)); \
break;

#define DEFAULT_DISPATCH(CLASS) void Visit##CLASS##Decl(CLASS##Decl *D) {}
#define DEFAULT_DISPATCH_VARDECL(CLASS) void Visit##CLASS##Decl(CLASS##Decl *D)\
  { static_cast<ImplClass*>(this)->VisitVarDecl(D); }


namespace clang {
template <typename ImplClass>
class CFGRecStmtDeclVisitor : public CFGRecStmtVisitor<ImplClass> {
public:

  void VisitDeclRefExpr(DeclRefExpr *DR) {
    static_cast<ImplClass*>(this)->VisitDecl(DR->getDecl());
  }

  void VisitDeclStmt(DeclStmt *DS) {
    for (DeclStmt::decl_iterator DI = DS->decl_begin(), DE = DS->decl_end();
        DI != DE; ++DI) {
      Decl *D = *DI;
      static_cast<ImplClass*>(this)->VisitDecl(D);
      // Visit the initializer.
      if (VarDecl *VD = dyn_cast<VarDecl>(D))
        if (Expr *I = VD->getInit())
          static_cast<ImplClass*>(this)->Visit(I);
    }
  }

  void VisitDecl(Decl *D) {
    switch (D->getKind()) {
        DISPATCH_CASE(Function)
        DISPATCH_CASE(CXXMethod)
        DISPATCH_CASE(Var)
        DISPATCH_CASE(ParmVar)       // FIXME: (same)
        DISPATCH_CASE(ImplicitParam)
        DISPATCH_CASE(EnumConstant)
        DISPATCH_CASE(Typedef)
        DISPATCH_CASE(Record)    // FIXME: Refine.  VisitStructDecl?
        DISPATCH_CASE(CXXRecord)
        DISPATCH_CASE(Enum)
        DISPATCH_CASE(Field)
        DISPATCH_CASE(UsingDirective)
        DISPATCH_CASE(Using)
        DISPATCH_CASE(NamespaceAlias)
      default:
        llvm_unreachable("Subtype of ScopedDecl not handled.");
    }
  }

  DEFAULT_DISPATCH(Var)
  DEFAULT_DISPATCH(Function)
  DEFAULT_DISPATCH(CXXMethod)
  DEFAULT_DISPATCH_VARDECL(ParmVar)
  DEFAULT_DISPATCH(ImplicitParam)
  DEFAULT_DISPATCH(EnumConstant)
  DEFAULT_DISPATCH(Typedef)
  DEFAULT_DISPATCH(Record)
  DEFAULT_DISPATCH(Enum)
  DEFAULT_DISPATCH(Field)
  DEFAULT_DISPATCH(ObjCInterface)
  DEFAULT_DISPATCH(ObjCMethod)
  DEFAULT_DISPATCH(ObjCProtocol)
  DEFAULT_DISPATCH(ObjCCategory)
  DEFAULT_DISPATCH(UsingDirective)
  DEFAULT_DISPATCH(Using)
  DEFAULT_DISPATCH(NamespaceAlias)

  void VisitCXXRecordDecl(CXXRecordDecl *D) {
    static_cast<ImplClass*>(this)->VisitRecordDecl(D);
  }
};

} // end namespace clang

#undef DISPATCH_CASE
#undef DEFAULT_DISPATCH
#endif
