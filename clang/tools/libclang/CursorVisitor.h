//===- CursorVisitor.h - CursorVisitor interface ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIBCLANG_CURSORVISITOR_H
#define LLVM_CLANG_LIBCLANG_CURSORVISITOR_H

#include "Index_Internal.h"
#include "CXCursor.h"
#include "CXTranslationUnit.h"

#include "clang/AST/DeclVisitor.h"
#include "clang/AST/TypeLocVisitor.h"

namespace clang {
  class PreprocessingRecord;
  class ASTUnit;

namespace cxcursor {

class VisitorJob {
public:
  enum Kind { DeclVisitKind, StmtVisitKind, MemberExprPartsKind,
              TypeLocVisitKind, OverloadExprPartsKind,
              DeclRefExprPartsKind, LabelRefVisitKind,
              ExplicitTemplateArgsVisitKind,
              NestedNameSpecifierLocVisitKind,
              DeclarationNameInfoVisitKind,
              MemberRefVisitKind, SizeOfPackExprPartsKind,
              LambdaExprPartsKind, PostChildrenVisitKind };
protected:
  void *data[3];
  CXCursor parent;
  Kind K;
  VisitorJob(CXCursor C, Kind k, void *d1, void *d2 = 0, void *d3 = 0)
    : parent(C), K(k) {
    data[0] = d1;
    data[1] = d2;
    data[2] = d3;
  }
public:
  Kind getKind() const { return K; }
  const CXCursor &getParent() const { return parent; }
};
  
typedef SmallVector<VisitorJob, 10> VisitorWorkList;

// Cursor visitor.
class CursorVisitor : public DeclVisitor<CursorVisitor, bool>,
                      public TypeLocVisitor<CursorVisitor, bool>
{
public:
  /// \brief Callback called after child nodes of a cursor have been visited.
  /// Return true to break visitation or false to continue.
  typedef bool (*PostChildrenVisitorTy)(CXCursor cursor,
                                        CXClientData client_data);

private:
  /// \brief The translation unit we are traversing.
  CXTranslationUnit TU;
  ASTUnit *AU;

  /// \brief The parent cursor whose children we are traversing.
  CXCursor Parent;

  /// \brief The declaration that serves at the parent of any statement or
  /// expression nodes.
  Decl *StmtParent;

  /// \brief The visitor function.
  CXCursorVisitor Visitor;

  PostChildrenVisitorTy PostChildrenVisitor;

  /// \brief The opaque client data, to be passed along to the visitor.
  CXClientData ClientData;

  /// \brief Whether we should visit the preprocessing record entries last, 
  /// after visiting other declarations.
  bool VisitPreprocessorLast;

  /// \brief Whether we should visit declarations or preprocessing record
  /// entries that are #included inside the \arg RegionOfInterest.
  bool VisitIncludedEntities;
  
  /// \brief When valid, a source range to which the cursor should restrict
  /// its search.
  SourceRange RegionOfInterest;

  /// \brief Whether we should only visit declarations and not preprocessing
  /// record entries.
  bool VisitDeclsOnly;

  // FIXME: Eventually remove.  This part of a hack to support proper
  // iteration over all Decls contained lexically within an ObjC container.
  DeclContext::decl_iterator *DI_current;
  DeclContext::decl_iterator DE_current;
  SmallVectorImpl<Decl *>::iterator *FileDI_current;
  SmallVectorImpl<Decl *>::iterator FileDE_current;

  // Cache of pre-allocated worklists for data-recursion walk of Stmts.
  SmallVector<VisitorWorkList*, 5> WorkListFreeList;
  SmallVector<VisitorWorkList*, 5> WorkListCache;

  using DeclVisitor<CursorVisitor, bool>::Visit;
  using TypeLocVisitor<CursorVisitor, bool>::Visit;

  /// \brief Determine whether this particular source range comes before, comes
  /// after, or overlaps the region of interest.
  ///
  /// \param R a half-open source range retrieved from the abstract syntax tree.
  RangeComparisonResult CompareRegionOfInterest(SourceRange R);

  void visitDeclsFromFileRegion(FileID File, unsigned Offset, unsigned Length);

  class SetParentRAII {
    CXCursor &Parent;
    Decl *&StmtParent;
    CXCursor OldParent;

  public:
    SetParentRAII(CXCursor &Parent, Decl *&StmtParent, CXCursor NewParent)
      : Parent(Parent), StmtParent(StmtParent), OldParent(Parent)
    {
      Parent = NewParent;
      if (clang_isDeclaration(Parent.kind))
        StmtParent = getCursorDecl(Parent);
    }

    ~SetParentRAII() {
      Parent = OldParent;
      if (clang_isDeclaration(Parent.kind))
        StmtParent = getCursorDecl(Parent);
    }
  };

public:
  CursorVisitor(CXTranslationUnit TU, CXCursorVisitor Visitor,
                CXClientData ClientData,
                bool VisitPreprocessorLast,
                bool VisitIncludedPreprocessingEntries = false,
                SourceRange RegionOfInterest = SourceRange(),
                bool VisitDeclsOnly = false,
                PostChildrenVisitorTy PostChildrenVisitor = 0)
    : TU(TU), AU(static_cast<ASTUnit*>(TU->TUData)),
      Visitor(Visitor), PostChildrenVisitor(PostChildrenVisitor),
      ClientData(ClientData),
      VisitPreprocessorLast(VisitPreprocessorLast),
      VisitIncludedEntities(VisitIncludedPreprocessingEntries),
      RegionOfInterest(RegionOfInterest),
      VisitDeclsOnly(VisitDeclsOnly),
      DI_current(0), FileDI_current(0)
  {
    Parent.kind = CXCursor_NoDeclFound;
    Parent.data[0] = 0;
    Parent.data[1] = 0;
    Parent.data[2] = 0;
    StmtParent = 0;
  }

  ~CursorVisitor() {
    // Free the pre-allocated worklists for data-recursion.
    for (SmallVectorImpl<VisitorWorkList*>::iterator
          I = WorkListCache.begin(), E = WorkListCache.end(); I != E; ++I) {
      delete *I;
    }
  }

  ASTUnit *getASTUnit() const { return static_cast<ASTUnit*>(TU->TUData); }
  CXTranslationUnit getTU() const { return TU; }

  bool Visit(CXCursor Cursor, bool CheckedRegionOfInterest = false);

  /// \brief Visit declarations and preprocessed entities for the file region
  /// designated by \see RegionOfInterest.
  void visitFileRegion();
  
  bool visitPreprocessedEntitiesInRegion();

  bool shouldVisitIncludedEntities() const {
    return VisitIncludedEntities;
  }

  template<typename InputIterator>
  bool visitPreprocessedEntities(InputIterator First, InputIterator Last,
                                 PreprocessingRecord &PPRec,
                                 FileID FID = FileID());

  bool VisitChildren(CXCursor Parent);

  // Declaration visitors
  bool VisitTypeAliasDecl(TypeAliasDecl *D);
  bool VisitAttributes(Decl *D);
  bool VisitBlockDecl(BlockDecl *B);
  bool VisitCXXRecordDecl(CXXRecordDecl *D);
  llvm::Optional<bool> shouldVisitCursor(CXCursor C);
  bool VisitDeclContext(DeclContext *DC);
  bool VisitTranslationUnitDecl(TranslationUnitDecl *D);
  bool VisitTypedefDecl(TypedefDecl *D);
  bool VisitTagDecl(TagDecl *D);
  bool VisitClassTemplateSpecializationDecl(ClassTemplateSpecializationDecl *D);
  bool VisitClassTemplatePartialSpecializationDecl(
                                     ClassTemplatePartialSpecializationDecl *D);
  bool VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D);
  bool VisitEnumConstantDecl(EnumConstantDecl *D);
  bool VisitDeclaratorDecl(DeclaratorDecl *DD);
  bool VisitFunctionDecl(FunctionDecl *ND);
  bool VisitFieldDecl(FieldDecl *D);
  bool VisitVarDecl(VarDecl *);
  bool VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D);
  bool VisitFunctionTemplateDecl(FunctionTemplateDecl *D);
  bool VisitClassTemplateDecl(ClassTemplateDecl *D);
  bool VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D);
  bool VisitObjCMethodDecl(ObjCMethodDecl *ND);
  bool VisitObjCContainerDecl(ObjCContainerDecl *D);
  bool VisitObjCCategoryDecl(ObjCCategoryDecl *ND);
  bool VisitObjCProtocolDecl(ObjCProtocolDecl *PID);
  bool VisitObjCPropertyDecl(ObjCPropertyDecl *PD);
  bool VisitObjCInterfaceDecl(ObjCInterfaceDecl *D);
  bool VisitObjCImplDecl(ObjCImplDecl *D);
  bool VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D);
  bool VisitObjCImplementationDecl(ObjCImplementationDecl *D);
  // FIXME: ObjCCompatibleAliasDecl requires aliased-class locations.
  bool VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *PD);
  bool VisitLinkageSpecDecl(LinkageSpecDecl *D);
  bool VisitNamespaceDecl(NamespaceDecl *D);
  bool VisitNamespaceAliasDecl(NamespaceAliasDecl *D);
  bool VisitUsingDirectiveDecl(UsingDirectiveDecl *D);
  bool VisitUsingDecl(UsingDecl *D);
  bool VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D);
  bool VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D);
  
  // Name visitor
  bool VisitDeclarationNameInfo(DeclarationNameInfo Name);
  bool VisitNestedNameSpecifier(NestedNameSpecifier *NNS, SourceRange Range);
  bool VisitNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS);
  
  // Template visitors
  bool VisitTemplateParameters(const TemplateParameterList *Params);
  bool VisitTemplateName(TemplateName Name, SourceLocation Loc);
  bool VisitTemplateArgumentLoc(const TemplateArgumentLoc &TAL);
  
  // Type visitors
#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT) \
  bool Visit##CLASS##TypeLoc(CLASS##TypeLoc TyLoc);
#include "clang/AST/TypeLocNodes.def"

  bool VisitTagTypeLoc(TagTypeLoc TL);
  bool VisitArrayTypeLoc(ArrayTypeLoc TL);
  bool VisitFunctionTypeLoc(FunctionTypeLoc TL, bool SkipResultType = false);

  // Data-recursive visitor functions.
  bool IsInRegionOfInterest(CXCursor C);
  bool RunVisitorWorkList(VisitorWorkList &WL);
  void EnqueueWorkList(VisitorWorkList &WL, Stmt *S);
  LLVM_ATTRIBUTE_NOINLINE bool Visit(Stmt *S);
};

}
}

#endif

