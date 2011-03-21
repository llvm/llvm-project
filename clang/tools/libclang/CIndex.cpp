//===- CIndex.cpp - Clang-C Source Indexing Library -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the main API hooks in the Clang-C Source Indexing
// library.
//
//===----------------------------------------------------------------------===//

#include "CIndexer.h"
#include "CXCursor.h"
#include "CXTranslationUnit.h"
#include "CXString.h"
#include "CXType.h"
#include "CXSourceLocation.h"
#include "CIndexDiagnostic.h"

#include "clang/Basic/Version.h"

#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TypeLocVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Optional.h"
#include "clang/Analysis/Support/SaveAndRestore.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/Compiler.h"

using namespace clang;
using namespace clang::cxcursor;
using namespace clang::cxstring;

static CXTranslationUnit MakeCXTranslationUnit(ASTUnit *TU) {
  if (!TU)
    return 0;
  CXTranslationUnit D = new CXTranslationUnitImpl();
  D->TUData = TU;
  D->StringPool = createCXStringPool();
  return D;
}

/// \brief The result of comparing two source ranges.
enum RangeComparisonResult {
  /// \brief Either the ranges overlap or one of the ranges is invalid.
  RangeOverlap,

  /// \brief The first range ends before the second range starts.
  RangeBefore,

  /// \brief The first range starts after the second range ends.
  RangeAfter
};

/// \brief Compare two source ranges to determine their relative position in
/// the translation unit.
static RangeComparisonResult RangeCompare(SourceManager &SM,
                                          SourceRange R1,
                                          SourceRange R2) {
  assert(R1.isValid() && "First range is invalid?");
  assert(R2.isValid() && "Second range is invalid?");
  if (R1.getEnd() != R2.getBegin() &&
      SM.isBeforeInTranslationUnit(R1.getEnd(), R2.getBegin()))
    return RangeBefore;
  if (R2.getEnd() != R1.getBegin() &&
      SM.isBeforeInTranslationUnit(R2.getEnd(), R1.getBegin()))
    return RangeAfter;
  return RangeOverlap;
}

/// \brief Determine if a source location falls within, before, or after a
///   a given source range.
static RangeComparisonResult LocationCompare(SourceManager &SM,
                                             SourceLocation L, SourceRange R) {
  assert(R.isValid() && "First range is invalid?");
  assert(L.isValid() && "Second range is invalid?");
  if (L == R.getBegin() || L == R.getEnd())
    return RangeOverlap;
  if (SM.isBeforeInTranslationUnit(L, R.getBegin()))
    return RangeBefore;
  if (SM.isBeforeInTranslationUnit(R.getEnd(), L))
    return RangeAfter;
  return RangeOverlap;
}

/// \brief Translate a Clang source range into a CIndex source range.
///
/// Clang internally represents ranges where the end location points to the
/// start of the token at the end. However, for external clients it is more
/// useful to have a CXSourceRange be a proper half-open interval. This routine
/// does the appropriate translation.
CXSourceRange cxloc::translateSourceRange(const SourceManager &SM,
                                          const LangOptions &LangOpts,
                                          const CharSourceRange &R) {
  // We want the last character in this location, so we will adjust the
  // location accordingly.
  SourceLocation EndLoc = R.getEnd();
  if (EndLoc.isValid() && EndLoc.isMacroID())
    EndLoc = SM.getSpellingLoc(EndLoc);
  if (R.isTokenRange() && !EndLoc.isInvalid() && EndLoc.isFileID()) {
    unsigned Length = Lexer::MeasureTokenLength(EndLoc, SM, LangOpts);
    EndLoc = EndLoc.getFileLocWithOffset(Length);
  }

  CXSourceRange Result = { { (void *)&SM, (void *)&LangOpts },
                           R.getBegin().getRawEncoding(),
                           EndLoc.getRawEncoding() };
  return Result;
}

//===----------------------------------------------------------------------===//
// Cursor visitor.
//===----------------------------------------------------------------------===//

namespace {
  
class VisitorJob {
public:
  enum Kind { DeclVisitKind, StmtVisitKind, MemberExprPartsKind,
              TypeLocVisitKind, OverloadExprPartsKind,
              DeclRefExprPartsKind, LabelRefVisitKind,
              ExplicitTemplateArgsVisitKind,
              NestedNameSpecifierVisitKind,
              NestedNameSpecifierLocVisitKind,
              DeclarationNameInfoVisitKind,
              MemberRefVisitKind, SizeOfPackExprPartsKind };
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
  static bool classof(VisitorJob *VJ) { return true; }
};
  
typedef llvm::SmallVector<VisitorJob, 10> VisitorWorkList;

// Cursor visitor.
class CursorVisitor : public DeclVisitor<CursorVisitor, bool>,
                      public TypeLocVisitor<CursorVisitor, bool>
{
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

  /// \brief The opaque client data, to be passed along to the visitor.
  CXClientData ClientData;

  // MaxPCHLevel - the maximum PCH level of declarations that we will pass on
  // to the visitor. Declarations with a PCH level greater than this value will
  // be suppressed.
  unsigned MaxPCHLevel;

  /// \brief When valid, a source range to which the cursor should restrict
  /// its search.
  SourceRange RegionOfInterest;

  // FIXME: Eventually remove.  This part of a hack to support proper
  // iteration over all Decls contained lexically within an ObjC container.
  DeclContext::decl_iterator *DI_current;
  DeclContext::decl_iterator DE_current;

  // Cache of pre-allocated worklists for data-recursion walk of Stmts.
  llvm::SmallVector<VisitorWorkList*, 5> WorkListFreeList;
  llvm::SmallVector<VisitorWorkList*, 5> WorkListCache;

  using DeclVisitor<CursorVisitor, bool>::Visit;
  using TypeLocVisitor<CursorVisitor, bool>::Visit;

  /// \brief Determine whether this particular source range comes before, comes
  /// after, or overlaps the region of interest.
  ///
  /// \param R a half-open source range retrieved from the abstract syntax tree.
  RangeComparisonResult CompareRegionOfInterest(SourceRange R);

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
                unsigned MaxPCHLevel,
                SourceRange RegionOfInterest = SourceRange())
    : TU(TU), AU(static_cast<ASTUnit*>(TU->TUData)),
      Visitor(Visitor), ClientData(ClientData),
      MaxPCHLevel(MaxPCHLevel), RegionOfInterest(RegionOfInterest), 
      DI_current(0)
  {
    Parent.kind = CXCursor_NoDeclFound;
    Parent.data[0] = 0;
    Parent.data[1] = 0;
    Parent.data[2] = 0;
    StmtParent = 0;
  }

  ~CursorVisitor() {
    // Free the pre-allocated worklists for data-recursion.
    for (llvm::SmallVectorImpl<VisitorWorkList*>::iterator
          I = WorkListCache.begin(), E = WorkListCache.end(); I != E; ++I) {
      delete *I;
    }
  }

  ASTUnit *getASTUnit() const { return static_cast<ASTUnit*>(TU->TUData); }
  CXTranslationUnit getTU() const { return TU; }

  bool Visit(CXCursor Cursor, bool CheckedRegionOfInterest = false);
  
  std::pair<PreprocessingRecord::iterator, PreprocessingRecord::iterator>
    getPreprocessedEntities();

  bool VisitChildren(CXCursor Parent);

  // Declaration visitors
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
  bool VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *D);
  bool VisitObjCClassDecl(ObjCClassDecl *D);
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
  bool VisitQualifiedTypeLoc(QualifiedTypeLoc TL);
  bool VisitBuiltinTypeLoc(BuiltinTypeLoc TL);
  bool VisitTypedefTypeLoc(TypedefTypeLoc TL);
  bool VisitUnresolvedUsingTypeLoc(UnresolvedUsingTypeLoc TL);
  bool VisitTagTypeLoc(TagTypeLoc TL);
  bool VisitTemplateTypeParmTypeLoc(TemplateTypeParmTypeLoc TL);
  bool VisitObjCInterfaceTypeLoc(ObjCInterfaceTypeLoc TL);
  bool VisitObjCObjectTypeLoc(ObjCObjectTypeLoc TL);
  bool VisitObjCObjectPointerTypeLoc(ObjCObjectPointerTypeLoc TL);
  bool VisitParenTypeLoc(ParenTypeLoc TL);
  bool VisitPointerTypeLoc(PointerTypeLoc TL);
  bool VisitBlockPointerTypeLoc(BlockPointerTypeLoc TL);
  bool VisitMemberPointerTypeLoc(MemberPointerTypeLoc TL);
  bool VisitLValueReferenceTypeLoc(LValueReferenceTypeLoc TL);
  bool VisitRValueReferenceTypeLoc(RValueReferenceTypeLoc TL);
  bool VisitFunctionTypeLoc(FunctionTypeLoc TL, bool SkipResultType = false);
  bool VisitArrayTypeLoc(ArrayTypeLoc TL);
  bool VisitTemplateSpecializationTypeLoc(TemplateSpecializationTypeLoc TL);
  // FIXME: Implement visitors here when the unimplemented TypeLocs get
  // implemented
  bool VisitTypeOfExprTypeLoc(TypeOfExprTypeLoc TL);
  bool VisitPackExpansionTypeLoc(PackExpansionTypeLoc TL);
  bool VisitTypeOfTypeLoc(TypeOfTypeLoc TL);
  bool VisitDependentNameTypeLoc(DependentNameTypeLoc TL);
  bool VisitDependentTemplateSpecializationTypeLoc(
                                    DependentTemplateSpecializationTypeLoc TL);
  bool VisitElaboratedTypeLoc(ElaboratedTypeLoc TL);
  
  // Data-recursive visitor functions.
  bool IsInRegionOfInterest(CXCursor C);
  bool RunVisitorWorkList(VisitorWorkList &WL);
  void EnqueueWorkList(VisitorWorkList &WL, Stmt *S);
  LLVM_ATTRIBUTE_NOINLINE bool Visit(Stmt *S);
};

} // end anonymous namespace

static SourceRange getRawCursorExtent(CXCursor C);
static SourceRange getFullCursorExtent(CXCursor C, SourceManager &SrcMgr);


RangeComparisonResult CursorVisitor::CompareRegionOfInterest(SourceRange R) {
  return RangeCompare(AU->getSourceManager(), R, RegionOfInterest);
}

/// \brief Visit the given cursor and, if requested by the visitor,
/// its children.
///
/// \param Cursor the cursor to visit.
///
/// \param CheckRegionOfInterest if true, then the caller already checked that
/// this cursor is within the region of interest.
///
/// \returns true if the visitation should be aborted, false if it
/// should continue.
bool CursorVisitor::Visit(CXCursor Cursor, bool CheckedRegionOfInterest) {
  if (clang_isInvalid(Cursor.kind))
    return false;

  if (clang_isDeclaration(Cursor.kind)) {
    Decl *D = getCursorDecl(Cursor);
    assert(D && "Invalid declaration cursor");
    if (D->getPCHLevel() > MaxPCHLevel)
      return false;

    if (D->isImplicit())
      return false;
  }

  // If we have a range of interest, and this cursor doesn't intersect with it,
  // we're done.
  if (RegionOfInterest.isValid() && !CheckedRegionOfInterest) {
    SourceRange Range = getRawCursorExtent(Cursor);
    if (Range.isInvalid() || CompareRegionOfInterest(Range))
      return false;
  }

  switch (Visitor(Cursor, Parent, ClientData)) {
  case CXChildVisit_Break:
    return true;

  case CXChildVisit_Continue:
    return false;

  case CXChildVisit_Recurse:
    return VisitChildren(Cursor);
  }

  return false;
}

std::pair<PreprocessingRecord::iterator, PreprocessingRecord::iterator>
CursorVisitor::getPreprocessedEntities() {
  PreprocessingRecord &PPRec
    = *AU->getPreprocessor().getPreprocessingRecord();
  
  bool OnlyLocalDecls
    = !AU->isMainFileAST() && AU->getOnlyLocalDecls(); 
  
  if (OnlyLocalDecls && RegionOfInterest.isValid()) {
    // If we would only look at local declarations but we have a region of 
    // interest, check whether that region of interest is in the main file.
    // If not, we should traverse all declarations.
    // FIXME: My kingdom for a proper binary search approach to finding
    // cursors!
    std::pair<FileID, unsigned> Location
      = AU->getSourceManager().getDecomposedInstantiationLoc(
                                                   RegionOfInterest.getBegin());
    if (Location.first != AU->getSourceManager().getMainFileID())
      OnlyLocalDecls = false;
  }
  
  PreprocessingRecord::iterator StartEntity, EndEntity;
  if (OnlyLocalDecls) {
    StartEntity = AU->pp_entity_begin();
    EndEntity = AU->pp_entity_end();
  } else {
    StartEntity = PPRec.begin();
    EndEntity = PPRec.end();
  }
  
  // There is no region of interest; we have to walk everything.
  if (RegionOfInterest.isInvalid())
    return std::make_pair(StartEntity, EndEntity);

  // Find the file in which the region of interest lands.
  SourceManager &SM = AU->getSourceManager();
  std::pair<FileID, unsigned> Begin
    = SM.getDecomposedInstantiationLoc(RegionOfInterest.getBegin());
  std::pair<FileID, unsigned> End
    = SM.getDecomposedInstantiationLoc(RegionOfInterest.getEnd());
  
  // The region of interest spans files; we have to walk everything.
  if (Begin.first != End.first)
    return std::make_pair(StartEntity, EndEntity);
    
  ASTUnit::PreprocessedEntitiesByFileMap &ByFileMap
    = AU->getPreprocessedEntitiesByFile();
  if (ByFileMap.empty()) {
    // Build the mapping from files to sets of preprocessed entities.
    for (PreprocessingRecord::iterator E = StartEntity; E != EndEntity; ++E) {
      std::pair<FileID, unsigned> P
        = SM.getDecomposedInstantiationLoc((*E)->getSourceRange().getBegin());
      
      ByFileMap[P.first].push_back(*E);
    }
  }

  return std::make_pair(ByFileMap[Begin.first].begin(), 
                        ByFileMap[Begin.first].end());
}

/// \brief Visit the children of the given cursor.
/// 
/// \returns true if the visitation should be aborted, false if it
/// should continue.
bool CursorVisitor::VisitChildren(CXCursor Cursor) {
  if (clang_isReference(Cursor.kind) && 
      Cursor.kind != CXCursor_CXXBaseSpecifier) {
    // By definition, references have no children.
    return false;
  }

  // Set the Parent field to Cursor, then back to its old value once we're
  // done.
  SetParentRAII SetParent(Parent, StmtParent, Cursor);

  if (clang_isDeclaration(Cursor.kind)) {
    Decl *D = getCursorDecl(Cursor);
    assert(D && "Invalid declaration cursor");
    return VisitAttributes(D) || Visit(D);
  }

  if (clang_isStatement(Cursor.kind))
    return Visit(getCursorStmt(Cursor));
  if (clang_isExpression(Cursor.kind))
    return Visit(getCursorExpr(Cursor));

  if (clang_isTranslationUnit(Cursor.kind)) {
    CXTranslationUnit tu = getCursorTU(Cursor);
    ASTUnit *CXXUnit = static_cast<ASTUnit*>(tu->TUData);
    if (!CXXUnit->isMainFileAST() && CXXUnit->getOnlyLocalDecls() &&
        RegionOfInterest.isInvalid()) {
      for (ASTUnit::top_level_iterator TL = CXXUnit->top_level_begin(),
                                    TLEnd = CXXUnit->top_level_end();
           TL != TLEnd; ++TL) {
        if (Visit(MakeCXCursor(*TL, tu), true))
          return true;
      }
    } else if (VisitDeclContext(
                            CXXUnit->getASTContext().getTranslationUnitDecl()))
      return true;

    // Walk the preprocessing record.
    if (CXXUnit->getPreprocessor().getPreprocessingRecord()) {
      // FIXME: Once we have the ability to deserialize a preprocessing record,
      // do so.
      PreprocessingRecord::iterator E, EEnd;
      for (llvm::tie(E, EEnd) = getPreprocessedEntities(); E != EEnd; ++E) {
        if (MacroInstantiation *MI = dyn_cast<MacroInstantiation>(*E)) {
          if (Visit(MakeMacroInstantiationCursor(MI, tu)))
            return true;
          
          continue;
        }
        
        if (MacroDefinition *MD = dyn_cast<MacroDefinition>(*E)) {
          if (Visit(MakeMacroDefinitionCursor(MD, tu)))
            return true;
          
          continue;
        }
        
        if (InclusionDirective *ID = dyn_cast<InclusionDirective>(*E)) {
          if (Visit(MakeInclusionDirectiveCursor(ID, tu)))
            return true;
          
          continue;
        }
      }
    }
    return false;
  }

  if (Cursor.kind == CXCursor_CXXBaseSpecifier) {
    if (CXXBaseSpecifier *Base = getCursorCXXBaseSpecifier(Cursor)) {
      if (TypeSourceInfo *BaseTSInfo = Base->getTypeSourceInfo()) {
        return Visit(BaseTSInfo->getTypeLoc());
      }
    }
  }
  
  // Nothing to visit at the moment.
  return false;
}

bool CursorVisitor::VisitBlockDecl(BlockDecl *B) {
  if (Visit(B->getSignatureAsWritten()->getTypeLoc()))
    return true;

  if (Stmt *Body = B->getBody())
    return Visit(MakeCXCursor(Body, StmtParent, TU));

  return false;
}

llvm::Optional<bool> CursorVisitor::shouldVisitCursor(CXCursor Cursor) {
  if (RegionOfInterest.isValid()) {
    SourceRange Range = getFullCursorExtent(Cursor, AU->getSourceManager());
    if (Range.isInvalid())
      return llvm::Optional<bool>();
    
    switch (CompareRegionOfInterest(Range)) {
    case RangeBefore:
      // This declaration comes before the region of interest; skip it.
      return llvm::Optional<bool>();

    case RangeAfter:
      // This declaration comes after the region of interest; we're done.
      return false;

    case RangeOverlap:
      // This declaration overlaps the region of interest; visit it.
      break;
    }
  }
  return true;
}

bool CursorVisitor::VisitDeclContext(DeclContext *DC) {
  DeclContext::decl_iterator I = DC->decls_begin(), E = DC->decls_end();

  // FIXME: Eventually remove.  This part of a hack to support proper
  // iteration over all Decls contained lexically within an ObjC container.
  SaveAndRestore<DeclContext::decl_iterator*> DI_saved(DI_current, &I);
  SaveAndRestore<DeclContext::decl_iterator> DE_saved(DE_current, E);

  for ( ; I != E; ++I) {
    Decl *D = *I;
    if (D->getLexicalDeclContext() != DC)
      continue;
    CXCursor Cursor = MakeCXCursor(D, TU);
    const llvm::Optional<bool> &V = shouldVisitCursor(Cursor);
    if (!V.hasValue())
      continue;
    if (!V.getValue())
      return false;
    if (Visit(Cursor, true))
      return true;
  }
  return false;
}

bool CursorVisitor::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
  llvm_unreachable("Translation units are visited directly by Visit()");
  return false;
}

bool CursorVisitor::VisitTypedefDecl(TypedefDecl *D) {
  if (TypeSourceInfo *TSInfo = D->getTypeSourceInfo())
    return Visit(TSInfo->getTypeLoc());

  return false;
}

bool CursorVisitor::VisitTagDecl(TagDecl *D) {
  return VisitDeclContext(D);
}

bool CursorVisitor::VisitClassTemplateSpecializationDecl(
                                          ClassTemplateSpecializationDecl *D) {
  bool ShouldVisitBody = false;
  switch (D->getSpecializationKind()) {
  case TSK_Undeclared:
  case TSK_ImplicitInstantiation:
    // Nothing to visit
    return false;
      
  case TSK_ExplicitInstantiationDeclaration:
  case TSK_ExplicitInstantiationDefinition:
    break;
      
  case TSK_ExplicitSpecialization:
    ShouldVisitBody = true;
    break;
  }
  
  // Visit the template arguments used in the specialization.
  if (TypeSourceInfo *SpecType = D->getTypeAsWritten()) {
    TypeLoc TL = SpecType->getTypeLoc();
    if (TemplateSpecializationTypeLoc *TSTLoc
          = dyn_cast<TemplateSpecializationTypeLoc>(&TL)) {
      for (unsigned I = 0, N = TSTLoc->getNumArgs(); I != N; ++I)
        if (VisitTemplateArgumentLoc(TSTLoc->getArgLoc(I)))
          return true;
    }
  }
  
  if (ShouldVisitBody && VisitCXXRecordDecl(D))
    return true;
  
  return false;
}

bool CursorVisitor::VisitClassTemplatePartialSpecializationDecl(
                                   ClassTemplatePartialSpecializationDecl *D) {
  // FIXME: Visit the "outer" template parameter lists on the TagDecl
  // before visiting these template parameters.
  if (VisitTemplateParameters(D->getTemplateParameters()))
    return true;

  // Visit the partial specialization arguments.
  const TemplateArgumentLoc *TemplateArgs = D->getTemplateArgsAsWritten();
  for (unsigned I = 0, N = D->getNumTemplateArgsAsWritten(); I != N; ++I)
    if (VisitTemplateArgumentLoc(TemplateArgs[I]))
      return true;
  
  return VisitCXXRecordDecl(D);
}

bool CursorVisitor::VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D) {
  // Visit the default argument.
  if (D->hasDefaultArgument() && !D->defaultArgumentWasInherited())
    if (TypeSourceInfo *DefArg = D->getDefaultArgumentInfo())
      if (Visit(DefArg->getTypeLoc()))
        return true;
  
  return false;
}

bool CursorVisitor::VisitEnumConstantDecl(EnumConstantDecl *D) {
  if (Expr *Init = D->getInitExpr())
    return Visit(MakeCXCursor(Init, StmtParent, TU));
  return false;
}

bool CursorVisitor::VisitDeclaratorDecl(DeclaratorDecl *DD) {
  if (TypeSourceInfo *TSInfo = DD->getTypeSourceInfo())
    if (Visit(TSInfo->getTypeLoc()))
      return true;

  // Visit the nested-name-specifier, if present.
  if (NestedNameSpecifierLoc QualifierLoc = DD->getQualifierLoc())
    if (VisitNestedNameSpecifierLoc(QualifierLoc))
      return true;

  return false;
}

/// \brief Compare two base or member initializers based on their source order.
static int CompareCXXCtorInitializers(const void* Xp, const void *Yp) {
  CXXCtorInitializer const * const *X
    = static_cast<CXXCtorInitializer const * const *>(Xp);
  CXXCtorInitializer const * const *Y
    = static_cast<CXXCtorInitializer const * const *>(Yp);
  
  if ((*X)->getSourceOrder() < (*Y)->getSourceOrder())
    return -1;
  else if ((*X)->getSourceOrder() > (*Y)->getSourceOrder())
    return 1;
  else
    return 0;
}

bool CursorVisitor::VisitFunctionDecl(FunctionDecl *ND) {
  if (TypeSourceInfo *TSInfo = ND->getTypeSourceInfo()) {
    // Visit the function declaration's syntactic components in the order
    // written. This requires a bit of work.
    TypeLoc TL = TSInfo->getTypeLoc().IgnoreParens();
    FunctionTypeLoc *FTL = dyn_cast<FunctionTypeLoc>(&TL);
    
    // If we have a function declared directly (without the use of a typedef),
    // visit just the return type. Otherwise, just visit the function's type
    // now.
    if ((FTL && !isa<CXXConversionDecl>(ND) && Visit(FTL->getResultLoc())) ||
        (!FTL && Visit(TL)))
      return true;
    
    // Visit the nested-name-specifier, if present.
    if (NestedNameSpecifierLoc QualifierLoc = ND->getQualifierLoc())
      if (VisitNestedNameSpecifierLoc(QualifierLoc))
        return true;
    
    // Visit the declaration name.
    if (VisitDeclarationNameInfo(ND->getNameInfo()))
      return true;
    
    // FIXME: Visit explicitly-specified template arguments!
    
    // Visit the function parameters, if we have a function type.
    if (FTL && VisitFunctionTypeLoc(*FTL, true))
      return true;
    
    // FIXME: Attributes?
  }
  
  if (ND->isThisDeclarationADefinition()) {
    if (CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(ND)) {
      // Find the initializers that were written in the source.
      llvm::SmallVector<CXXCtorInitializer *, 4> WrittenInits;
      for (CXXConstructorDecl::init_iterator I = Constructor->init_begin(),
                                          IEnd = Constructor->init_end();
           I != IEnd; ++I) {
        if (!(*I)->isWritten())
          continue;
      
        WrittenInits.push_back(*I);
      }
      
      // Sort the initializers in source order
      llvm::array_pod_sort(WrittenInits.begin(), WrittenInits.end(),
                           &CompareCXXCtorInitializers);
      
      // Visit the initializers in source order
      for (unsigned I = 0, N = WrittenInits.size(); I != N; ++I) {
        CXXCtorInitializer *Init = WrittenInits[I];
        if (Init->isAnyMemberInitializer()) {
          if (Visit(MakeCursorMemberRef(Init->getAnyMember(),
                                        Init->getMemberLocation(), TU)))
            return true;
        } else if (TypeSourceInfo *BaseInfo = Init->getBaseClassInfo()) {
          if (Visit(BaseInfo->getTypeLoc()))
            return true;
        }
        
        // Visit the initializer value.
        if (Expr *Initializer = Init->getInit())
          if (Visit(MakeCXCursor(Initializer, ND, TU)))
            return true;
      } 
    }
    
    if (Visit(MakeCXCursor(ND->getBody(), StmtParent, TU)))
      return true;
  }

  return false;
}

bool CursorVisitor::VisitFieldDecl(FieldDecl *D) {
  if (VisitDeclaratorDecl(D))
    return true;

  if (Expr *BitWidth = D->getBitWidth())
    return Visit(MakeCXCursor(BitWidth, StmtParent, TU));

  return false;
}

bool CursorVisitor::VisitVarDecl(VarDecl *D) {
  if (VisitDeclaratorDecl(D))
    return true;

  if (Expr *Init = D->getInit())
    return Visit(MakeCXCursor(Init, StmtParent, TU));

  return false;
}

bool CursorVisitor::VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D) {
  if (VisitDeclaratorDecl(D))
    return true;
  
  if (D->hasDefaultArgument() && !D->defaultArgumentWasInherited())
    if (Expr *DefArg = D->getDefaultArgument())
      return Visit(MakeCXCursor(DefArg, StmtParent, TU));
  
  return false;  
}

bool CursorVisitor::VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
  // FIXME: Visit the "outer" template parameter lists on the FunctionDecl
  // before visiting these template parameters.
  if (VisitTemplateParameters(D->getTemplateParameters()))
    return true;
  
  return VisitFunctionDecl(D->getTemplatedDecl());
}

bool CursorVisitor::VisitClassTemplateDecl(ClassTemplateDecl *D) {
  // FIXME: Visit the "outer" template parameter lists on the TagDecl
  // before visiting these template parameters.
  if (VisitTemplateParameters(D->getTemplateParameters()))
    return true;
  
  return VisitCXXRecordDecl(D->getTemplatedDecl());
}

bool CursorVisitor::VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D) {
  if (VisitTemplateParameters(D->getTemplateParameters()))
    return true;
  
  if (D->hasDefaultArgument() && !D->defaultArgumentWasInherited() &&
      VisitTemplateArgumentLoc(D->getDefaultArgument()))
    return true;
  
  return false;
}

bool CursorVisitor::VisitObjCMethodDecl(ObjCMethodDecl *ND) {
  if (TypeSourceInfo *TSInfo = ND->getResultTypeSourceInfo())
    if (Visit(TSInfo->getTypeLoc()))
      return true;

  for (ObjCMethodDecl::param_iterator P = ND->param_begin(),
       PEnd = ND->param_end();
       P != PEnd; ++P) {
    if (Visit(MakeCXCursor(*P, TU)))
      return true;
  }

  if (ND->isThisDeclarationADefinition() &&
      Visit(MakeCXCursor(ND->getBody(), StmtParent, TU)))
    return true;

  return false;
}

namespace {
  struct ContainerDeclsSort {
    SourceManager &SM;
    ContainerDeclsSort(SourceManager &sm) : SM(sm) {}
    bool operator()(Decl *A, Decl *B) {
      SourceLocation L_A = A->getLocStart();
      SourceLocation L_B = B->getLocStart();
      assert(L_A.isValid() && L_B.isValid());
      return SM.isBeforeInTranslationUnit(L_A, L_B);
    }
  };
}

bool CursorVisitor::VisitObjCContainerDecl(ObjCContainerDecl *D) {
  // FIXME: Eventually convert back to just 'VisitDeclContext()'.  Essentially
  // an @implementation can lexically contain Decls that are not properly
  // nested in the AST.  When we identify such cases, we need to retrofit
  // this nesting here.
  if (!DI_current)
    return VisitDeclContext(D);

  // Scan the Decls that immediately come after the container
  // in the current DeclContext.  If any fall within the
  // container's lexical region, stash them into a vector
  // for later processing.
  llvm::SmallVector<Decl *, 24> DeclsInContainer;
  SourceLocation EndLoc = D->getSourceRange().getEnd();
  SourceManager &SM = AU->getSourceManager();
  if (EndLoc.isValid()) {
    DeclContext::decl_iterator next = *DI_current;
    while (++next != DE_current) {
      Decl *D_next = *next;
      if (!D_next)
        break;
      SourceLocation L = D_next->getLocStart();
      if (!L.isValid())
        break;
      if (SM.isBeforeInTranslationUnit(L, EndLoc)) {
        *DI_current = next;
        DeclsInContainer.push_back(D_next);
        continue;
      }
      break;
    }
  }

  // The common case.
  if (DeclsInContainer.empty())
    return VisitDeclContext(D);

  // Get all the Decls in the DeclContext, and sort them with the
  // additional ones we've collected.  Then visit them.
  for (DeclContext::decl_iterator I = D->decls_begin(), E = D->decls_end();
       I!=E; ++I) {
    Decl *subDecl = *I;
    if (!subDecl || subDecl->getLexicalDeclContext() != D ||
        subDecl->getLocStart().isInvalid())
      continue;
    DeclsInContainer.push_back(subDecl);
  }

  // Now sort the Decls so that they appear in lexical order.
  std::sort(DeclsInContainer.begin(), DeclsInContainer.end(),
            ContainerDeclsSort(SM));

  // Now visit the decls.
  for (llvm::SmallVectorImpl<Decl*>::iterator I = DeclsInContainer.begin(),
         E = DeclsInContainer.end(); I != E; ++I) {
    CXCursor Cursor = MakeCXCursor(*I, TU);
    const llvm::Optional<bool> &V = shouldVisitCursor(Cursor);
    if (!V.hasValue())
      continue;
    if (!V.getValue())
      return false;
    if (Visit(Cursor, true))
      return true;
  }
  return false;
}

bool CursorVisitor::VisitObjCCategoryDecl(ObjCCategoryDecl *ND) {
  if (Visit(MakeCursorObjCClassRef(ND->getClassInterface(), ND->getLocation(),
                                   TU)))
    return true;

  ObjCCategoryDecl::protocol_loc_iterator PL = ND->protocol_loc_begin();
  for (ObjCCategoryDecl::protocol_iterator I = ND->protocol_begin(),
         E = ND->protocol_end(); I != E; ++I, ++PL)
    if (Visit(MakeCursorObjCProtocolRef(*I, *PL, TU)))
      return true;

  return VisitObjCContainerDecl(ND);
}

bool CursorVisitor::VisitObjCProtocolDecl(ObjCProtocolDecl *PID) {
  ObjCProtocolDecl::protocol_loc_iterator PL = PID->protocol_loc_begin();
  for (ObjCProtocolDecl::protocol_iterator I = PID->protocol_begin(),
       E = PID->protocol_end(); I != E; ++I, ++PL)
    if (Visit(MakeCursorObjCProtocolRef(*I, *PL, TU)))
      return true;

  return VisitObjCContainerDecl(PID);
}

bool CursorVisitor::VisitObjCPropertyDecl(ObjCPropertyDecl *PD) {
  if (PD->getTypeSourceInfo() && Visit(PD->getTypeSourceInfo()->getTypeLoc()))
    return true;

  // FIXME: This implements a workaround with @property declarations also being
  // installed in the DeclContext for the @interface.  Eventually this code
  // should be removed.
  ObjCCategoryDecl *CDecl = dyn_cast<ObjCCategoryDecl>(PD->getDeclContext());
  if (!CDecl || !CDecl->IsClassExtension())
    return false;

  ObjCInterfaceDecl *ID = CDecl->getClassInterface();
  if (!ID)
    return false;

  IdentifierInfo *PropertyId = PD->getIdentifier();
  ObjCPropertyDecl *prevDecl =
    ObjCPropertyDecl::findPropertyDecl(cast<DeclContext>(ID), PropertyId);

  if (!prevDecl)
    return false;

  // Visit synthesized methods since they will be skipped when visiting
  // the @interface.
  if (ObjCMethodDecl *MD = prevDecl->getGetterMethodDecl())
    if (MD->isSynthesized() && MD->getLexicalDeclContext() == CDecl)
      if (Visit(MakeCXCursor(MD, TU)))
        return true;

  if (ObjCMethodDecl *MD = prevDecl->getSetterMethodDecl())
    if (MD->isSynthesized() && MD->getLexicalDeclContext() == CDecl)
      if (Visit(MakeCXCursor(MD, TU)))
        return true;

  return false;
}

bool CursorVisitor::VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {
  // Issue callbacks for super class.
  if (D->getSuperClass() &&
      Visit(MakeCursorObjCSuperClassRef(D->getSuperClass(),
                                        D->getSuperClassLoc(),
                                        TU)))
    return true;

  ObjCInterfaceDecl::protocol_loc_iterator PL = D->protocol_loc_begin();
  for (ObjCInterfaceDecl::protocol_iterator I = D->protocol_begin(),
         E = D->protocol_end(); I != E; ++I, ++PL)
    if (Visit(MakeCursorObjCProtocolRef(*I, *PL, TU)))
      return true;

  return VisitObjCContainerDecl(D);
}

bool CursorVisitor::VisitObjCImplDecl(ObjCImplDecl *D) {
  return VisitObjCContainerDecl(D);
}

bool CursorVisitor::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
  // 'ID' could be null when dealing with invalid code.
  if (ObjCInterfaceDecl *ID = D->getClassInterface())
    if (Visit(MakeCursorObjCClassRef(ID, D->getLocation(), TU)))
      return true;

  return VisitObjCImplDecl(D);
}

bool CursorVisitor::VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
#if 0
  // Issue callbacks for super class.
  // FIXME: No source location information!
  if (D->getSuperClass() &&
      Visit(MakeCursorObjCSuperClassRef(D->getSuperClass(),
                                        D->getSuperClassLoc(),
                                        TU)))
    return true;
#endif

  return VisitObjCImplDecl(D);
}

bool CursorVisitor::VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *D) {
  ObjCForwardProtocolDecl::protocol_loc_iterator PL = D->protocol_loc_begin();
  for (ObjCForwardProtocolDecl::protocol_iterator I = D->protocol_begin(),
                                                  E = D->protocol_end();
       I != E; ++I, ++PL)
    if (Visit(MakeCursorObjCProtocolRef(*I, *PL, TU)))
      return true;

  return false;
}

bool CursorVisitor::VisitObjCClassDecl(ObjCClassDecl *D) {
  for (ObjCClassDecl::iterator C = D->begin(), CEnd = D->end(); C != CEnd; ++C)
    if (Visit(MakeCursorObjCClassRef(C->getInterface(), C->getLocation(), TU)))
      return true;

  return false;
}

bool CursorVisitor::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *PD) {
  if (ObjCIvarDecl *Ivar = PD->getPropertyIvarDecl())
    return Visit(MakeCursorMemberRef(Ivar, PD->getPropertyIvarDeclLoc(), TU));
  
  return false;
}

bool CursorVisitor::VisitNamespaceDecl(NamespaceDecl *D) {
  return VisitDeclContext(D);
}

bool CursorVisitor::VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {
  // Visit nested-name-specifier.
  if (NestedNameSpecifierLoc QualifierLoc = D->getQualifierLoc())
    if (VisitNestedNameSpecifierLoc(QualifierLoc))
      return true;
  
  return Visit(MakeCursorNamespaceRef(D->getAliasedNamespace(), 
                                      D->getTargetNameLoc(), TU));
}

bool CursorVisitor::VisitUsingDecl(UsingDecl *D) {
  // Visit nested-name-specifier.
  if (NestedNameSpecifierLoc QualifierLoc = D->getQualifierLoc()) {
    if (VisitNestedNameSpecifierLoc(QualifierLoc))
      return true;
  }
  
  if (Visit(MakeCursorOverloadedDeclRef(D, D->getLocation(), TU)))
    return true;
    
  return VisitDeclarationNameInfo(D->getNameInfo());
}

bool CursorVisitor::VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
  // Visit nested-name-specifier.
  if (NestedNameSpecifierLoc QualifierLoc = D->getQualifierLoc())
    if (VisitNestedNameSpecifierLoc(QualifierLoc))
      return true;

  return Visit(MakeCursorNamespaceRef(D->getNominatedNamespaceAsWritten(),
                                      D->getIdentLocation(), TU));
}

bool CursorVisitor::VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D) {
  // Visit nested-name-specifier.
  if (NestedNameSpecifierLoc QualifierLoc = D->getQualifierLoc()) {
    if (VisitNestedNameSpecifierLoc(QualifierLoc))
      return true;
  }

  return VisitDeclarationNameInfo(D->getNameInfo());
}

bool CursorVisitor::VisitUnresolvedUsingTypenameDecl(
                                               UnresolvedUsingTypenameDecl *D) {
  // Visit nested-name-specifier.
  if (NestedNameSpecifierLoc QualifierLoc = D->getQualifierLoc())
    if (VisitNestedNameSpecifierLoc(QualifierLoc))
      return true;
  
  return false;
}

bool CursorVisitor::VisitDeclarationNameInfo(DeclarationNameInfo Name) {
  switch (Name.getName().getNameKind()) {
  case clang::DeclarationName::Identifier:
  case clang::DeclarationName::CXXLiteralOperatorName:
  case clang::DeclarationName::CXXOperatorName:
  case clang::DeclarationName::CXXUsingDirective:
    return false;
      
  case clang::DeclarationName::CXXConstructorName:
  case clang::DeclarationName::CXXDestructorName:
  case clang::DeclarationName::CXXConversionFunctionName:
    if (TypeSourceInfo *TSInfo = Name.getNamedTypeInfo())
      return Visit(TSInfo->getTypeLoc());
    return false;

  case clang::DeclarationName::ObjCZeroArgSelector:
  case clang::DeclarationName::ObjCOneArgSelector:
  case clang::DeclarationName::ObjCMultiArgSelector:
    // FIXME: Per-identifier location info?
    return false;
  }
  
  return false;
}

bool CursorVisitor::VisitNestedNameSpecifier(NestedNameSpecifier *NNS, 
                                             SourceRange Range) {
  // FIXME: This whole routine is a hack to work around the lack of proper
  // source information in nested-name-specifiers (PR5791). Since we do have
  // a beginning source location, we can visit the first component of the
  // nested-name-specifier, if it's a single-token component.
  if (!NNS)
    return false;
  
  // Get the first component in the nested-name-specifier.
  while (NestedNameSpecifier *Prefix = NNS->getPrefix())
    NNS = Prefix;
  
  switch (NNS->getKind()) {
  case NestedNameSpecifier::Namespace:
    return Visit(MakeCursorNamespaceRef(NNS->getAsNamespace(), Range.getBegin(),
                                        TU));

  case NestedNameSpecifier::NamespaceAlias:
    return Visit(MakeCursorNamespaceRef(NNS->getAsNamespaceAlias(), 
                                        Range.getBegin(), TU));

  case NestedNameSpecifier::TypeSpec: {
    // If the type has a form where we know that the beginning of the source
    // range matches up with a reference cursor. Visit the appropriate reference
    // cursor.
    const Type *T = NNS->getAsType();
    if (const TypedefType *Typedef = dyn_cast<TypedefType>(T))
      return Visit(MakeCursorTypeRef(Typedef->getDecl(), Range.getBegin(), TU));
    if (const TagType *Tag = dyn_cast<TagType>(T))
      return Visit(MakeCursorTypeRef(Tag->getDecl(), Range.getBegin(), TU));
    if (const TemplateSpecializationType *TST
                                      = dyn_cast<TemplateSpecializationType>(T))
      return VisitTemplateName(TST->getTemplateName(), Range.getBegin());
    break;
  }
      
  case NestedNameSpecifier::TypeSpecWithTemplate:
  case NestedNameSpecifier::Global:
  case NestedNameSpecifier::Identifier:
    break;      
  }
  
  return false;
}

bool 
CursorVisitor::VisitNestedNameSpecifierLoc(NestedNameSpecifierLoc Qualifier) {
  llvm::SmallVector<NestedNameSpecifierLoc, 4> Qualifiers;
  for (; Qualifier; Qualifier = Qualifier.getPrefix())
    Qualifiers.push_back(Qualifier);
  
  while (!Qualifiers.empty()) {
    NestedNameSpecifierLoc Q = Qualifiers.pop_back_val();
    NestedNameSpecifier *NNS = Q.getNestedNameSpecifier();
    switch (NNS->getKind()) {
    case NestedNameSpecifier::Namespace:
      if (Visit(MakeCursorNamespaceRef(NNS->getAsNamespace(), 
                                       Q.getLocalBeginLoc(),
                                       TU)))
        return true;
        
      break;
      
    case NestedNameSpecifier::NamespaceAlias:
      if (Visit(MakeCursorNamespaceRef(NNS->getAsNamespaceAlias(), 
                                       Q.getLocalBeginLoc(),
                                       TU)))
        return true;
        
      break;
        
    case NestedNameSpecifier::TypeSpec:
    case NestedNameSpecifier::TypeSpecWithTemplate:
      if (Visit(Q.getTypeLoc()))
        return true;
        
      break;
        
    case NestedNameSpecifier::Global:
    case NestedNameSpecifier::Identifier:
      break;              
    }
  }
  
  return false;
}

bool CursorVisitor::VisitTemplateParameters(
                                          const TemplateParameterList *Params) {
  if (!Params)
    return false;
  
  for (TemplateParameterList::const_iterator P = Params->begin(),
                                          PEnd = Params->end();
       P != PEnd; ++P) {
    if (Visit(MakeCXCursor(*P, TU)))
      return true;
  }
  
  return false;
}

bool CursorVisitor::VisitTemplateName(TemplateName Name, SourceLocation Loc) {
  switch (Name.getKind()) {
  case TemplateName::Template:
    return Visit(MakeCursorTemplateRef(Name.getAsTemplateDecl(), Loc, TU));

  case TemplateName::OverloadedTemplate:
    // Visit the overloaded template set.
    if (Visit(MakeCursorOverloadedDeclRef(Name, Loc, TU)))
      return true;

    return false;

  case TemplateName::DependentTemplate:
    // FIXME: Visit nested-name-specifier.
    return false;
      
  case TemplateName::QualifiedTemplate:
    // FIXME: Visit nested-name-specifier.
    return Visit(MakeCursorTemplateRef(
                                  Name.getAsQualifiedTemplateName()->getDecl(), 
                                       Loc, TU));
      
  case TemplateName::SubstTemplateTemplateParmPack:
    return Visit(MakeCursorTemplateRef(
                  Name.getAsSubstTemplateTemplateParmPack()->getParameterPack(),
                                       Loc, TU));
  }
                 
  return false;
}

bool CursorVisitor::VisitTemplateArgumentLoc(const TemplateArgumentLoc &TAL) {
  switch (TAL.getArgument().getKind()) {
  case TemplateArgument::Null:
  case TemplateArgument::Integral:
  case TemplateArgument::Pack:
    return false;
      
  case TemplateArgument::Type:
    if (TypeSourceInfo *TSInfo = TAL.getTypeSourceInfo())
      return Visit(TSInfo->getTypeLoc());
    return false;
      
  case TemplateArgument::Declaration:
    if (Expr *E = TAL.getSourceDeclExpression())
      return Visit(MakeCXCursor(E, StmtParent, TU));
    return false;
      
  case TemplateArgument::Expression:
    if (Expr *E = TAL.getSourceExpression())
      return Visit(MakeCXCursor(E, StmtParent, TU));
    return false;
  
  case TemplateArgument::Template:
  case TemplateArgument::TemplateExpansion:
    if (VisitNestedNameSpecifierLoc(TAL.getTemplateQualifierLoc()))
      return true;
      
    return VisitTemplateName(TAL.getArgument().getAsTemplateOrTemplatePattern(), 
                             TAL.getTemplateNameLoc());
  }
  
  return false;
}

bool CursorVisitor::VisitLinkageSpecDecl(LinkageSpecDecl *D) {
  return VisitDeclContext(D);
}

bool CursorVisitor::VisitQualifiedTypeLoc(QualifiedTypeLoc TL) {
  return Visit(TL.getUnqualifiedLoc());
}

bool CursorVisitor::VisitBuiltinTypeLoc(BuiltinTypeLoc TL) {
  ASTContext &Context = AU->getASTContext();

  // Some builtin types (such as Objective-C's "id", "sel", and
  // "Class") have associated declarations. Create cursors for those.
  QualType VisitType;
  switch (TL.getType()->getAs<BuiltinType>()->getKind()) {
  case BuiltinType::Void:
  case BuiltinType::Bool:
  case BuiltinType::Char_U:
  case BuiltinType::UChar:
  case BuiltinType::Char16:
  case BuiltinType::Char32:
  case BuiltinType::UShort:
  case BuiltinType::UInt:
  case BuiltinType::ULong:
  case BuiltinType::ULongLong:
  case BuiltinType::UInt128:
  case BuiltinType::Char_S:
  case BuiltinType::SChar:
  case BuiltinType::WChar_U:
  case BuiltinType::WChar_S:
  case BuiltinType::Short:
  case BuiltinType::Int:
  case BuiltinType::Long:
  case BuiltinType::LongLong:
  case BuiltinType::Int128:
  case BuiltinType::Float:
  case BuiltinType::Double:
  case BuiltinType::LongDouble:
  case BuiltinType::NullPtr:
  case BuiltinType::Overload:
  case BuiltinType::Dependent:
    break;

  case BuiltinType::ObjCId:
    VisitType = Context.getObjCIdType();
    break;

  case BuiltinType::ObjCClass:
    VisitType = Context.getObjCClassType();
    break;

  case BuiltinType::ObjCSel:
    VisitType = Context.getObjCSelType();
    break;
  }

  if (!VisitType.isNull()) {
    if (const TypedefType *Typedef = VisitType->getAs<TypedefType>())
      return Visit(MakeCursorTypeRef(Typedef->getDecl(), TL.getBuiltinLoc(),
                                     TU));
  }

  return false;
}

bool CursorVisitor::VisitTypedefTypeLoc(TypedefTypeLoc TL) {
  return Visit(MakeCursorTypeRef(TL.getTypedefDecl(), TL.getNameLoc(), TU));
}

bool CursorVisitor::VisitUnresolvedUsingTypeLoc(UnresolvedUsingTypeLoc TL) {
  return Visit(MakeCursorTypeRef(TL.getDecl(), TL.getNameLoc(), TU));
}

bool CursorVisitor::VisitTagTypeLoc(TagTypeLoc TL) {
  return Visit(MakeCursorTypeRef(TL.getDecl(), TL.getNameLoc(), TU));
}

bool CursorVisitor::VisitTemplateTypeParmTypeLoc(TemplateTypeParmTypeLoc TL) {
  // FIXME: We can't visit the template type parameter, because there's
  // no context information with which we can match up the depth/index in the
  // type to the appropriate 
  return false;
}

bool CursorVisitor::VisitObjCInterfaceTypeLoc(ObjCInterfaceTypeLoc TL) {
  if (Visit(MakeCursorObjCClassRef(TL.getIFaceDecl(), TL.getNameLoc(), TU)))
    return true;

  return false;
}

bool CursorVisitor::VisitObjCObjectTypeLoc(ObjCObjectTypeLoc TL) {
  if (TL.hasBaseTypeAsWritten() && Visit(TL.getBaseLoc()))
    return true;

  for (unsigned I = 0, N = TL.getNumProtocols(); I != N; ++I) {
    if (Visit(MakeCursorObjCProtocolRef(TL.getProtocol(I), TL.getProtocolLoc(I),
                                        TU)))
      return true;
  }

  return false;
}

bool CursorVisitor::VisitObjCObjectPointerTypeLoc(ObjCObjectPointerTypeLoc TL) {
  return Visit(TL.getPointeeLoc());
}

bool CursorVisitor::VisitParenTypeLoc(ParenTypeLoc TL) {
  return Visit(TL.getInnerLoc());
}

bool CursorVisitor::VisitPointerTypeLoc(PointerTypeLoc TL) {
  return Visit(TL.getPointeeLoc());
}

bool CursorVisitor::VisitBlockPointerTypeLoc(BlockPointerTypeLoc TL) {
  return Visit(TL.getPointeeLoc());
}

bool CursorVisitor::VisitMemberPointerTypeLoc(MemberPointerTypeLoc TL) {
  return Visit(TL.getPointeeLoc());
}

bool CursorVisitor::VisitLValueReferenceTypeLoc(LValueReferenceTypeLoc TL) {
  return Visit(TL.getPointeeLoc());
}

bool CursorVisitor::VisitRValueReferenceTypeLoc(RValueReferenceTypeLoc TL) {
  return Visit(TL.getPointeeLoc());
}

bool CursorVisitor::VisitFunctionTypeLoc(FunctionTypeLoc TL, 
                                         bool SkipResultType) {
  if (!SkipResultType && Visit(TL.getResultLoc()))
    return true;

  for (unsigned I = 0, N = TL.getNumArgs(); I != N; ++I)
    if (Decl *D = TL.getArg(I))
      if (Visit(MakeCXCursor(D, TU)))
        return true;

  return false;
}

bool CursorVisitor::VisitArrayTypeLoc(ArrayTypeLoc TL) {
  if (Visit(TL.getElementLoc()))
    return true;

  if (Expr *Size = TL.getSizeExpr())
    return Visit(MakeCXCursor(Size, StmtParent, TU));

  return false;
}

bool CursorVisitor::VisitTemplateSpecializationTypeLoc(
                                             TemplateSpecializationTypeLoc TL) {
  // Visit the template name.
  if (VisitTemplateName(TL.getTypePtr()->getTemplateName(), 
                        TL.getTemplateNameLoc()))
    return true;
  
  // Visit the template arguments.
  for (unsigned I = 0, N = TL.getNumArgs(); I != N; ++I)
    if (VisitTemplateArgumentLoc(TL.getArgLoc(I)))
      return true;
  
  return false;
}

bool CursorVisitor::VisitTypeOfExprTypeLoc(TypeOfExprTypeLoc TL) {
  return Visit(MakeCXCursor(TL.getUnderlyingExpr(), StmtParent, TU));
}

bool CursorVisitor::VisitTypeOfTypeLoc(TypeOfTypeLoc TL) {
  if (TypeSourceInfo *TSInfo = TL.getUnderlyingTInfo())
    return Visit(TSInfo->getTypeLoc());

  return false;
}

bool CursorVisitor::VisitDependentNameTypeLoc(DependentNameTypeLoc TL) {
  if (VisitNestedNameSpecifierLoc(TL.getQualifierLoc()))
    return true;
  
  return false;
}

bool CursorVisitor::VisitDependentTemplateSpecializationTypeLoc(
                                    DependentTemplateSpecializationTypeLoc TL) {
  // Visit the nested-name-specifier, if there is one.
  if (TL.getQualifierLoc() &&
      VisitNestedNameSpecifierLoc(TL.getQualifierLoc()))
    return true;
  
  // Visit the template arguments.
  for (unsigned I = 0, N = TL.getNumArgs(); I != N; ++I)
    if (VisitTemplateArgumentLoc(TL.getArgLoc(I)))
      return true;

  return false;
}

bool CursorVisitor::VisitElaboratedTypeLoc(ElaboratedTypeLoc TL) {
  if (VisitNestedNameSpecifierLoc(TL.getQualifierLoc()))
    return true;
  
  return Visit(TL.getNamedTypeLoc());
}

bool CursorVisitor::VisitPackExpansionTypeLoc(PackExpansionTypeLoc TL) {
  return Visit(TL.getPatternLoc());
}

bool CursorVisitor::VisitCXXRecordDecl(CXXRecordDecl *D) {
  // Visit the nested-name-specifier, if present.
  if (NestedNameSpecifierLoc QualifierLoc = D->getQualifierLoc())
    if (VisitNestedNameSpecifierLoc(QualifierLoc))
      return true;

  if (D->isDefinition()) {
    for (CXXRecordDecl::base_class_iterator I = D->bases_begin(),
         E = D->bases_end(); I != E; ++I) {
      if (Visit(cxcursor::MakeCursorCXXBaseSpecifier(I, TU)))
        return true;
    }
  }

  return VisitTagDecl(D);
}

bool CursorVisitor::VisitAttributes(Decl *D) {
  for (AttrVec::const_iterator i = D->attr_begin(), e = D->attr_end();
       i != e; ++i)
    if (Visit(MakeCXCursor(*i, D, TU)))
        return true;

  return false;
}

//===----------------------------------------------------------------------===//
// Data-recursive visitor methods.
//===----------------------------------------------------------------------===//

namespace {
#define DEF_JOB(NAME, DATA, KIND)\
class NAME : public VisitorJob {\
public:\
  NAME(DATA *d, CXCursor parent) : VisitorJob(parent, VisitorJob::KIND, d) {} \
  static bool classof(const VisitorJob *VJ) { return VJ->getKind() == KIND; }\
  DATA *get() const { return static_cast<DATA*>(data[0]); }\
};

DEF_JOB(StmtVisit, Stmt, StmtVisitKind)
DEF_JOB(MemberExprParts, MemberExpr, MemberExprPartsKind)
DEF_JOB(DeclRefExprParts, DeclRefExpr, DeclRefExprPartsKind)
DEF_JOB(OverloadExprParts, OverloadExpr, OverloadExprPartsKind)
DEF_JOB(ExplicitTemplateArgsVisit, ExplicitTemplateArgumentList, 
        ExplicitTemplateArgsVisitKind)
DEF_JOB(SizeOfPackExprParts, SizeOfPackExpr, SizeOfPackExprPartsKind)
#undef DEF_JOB

class DeclVisit : public VisitorJob {
public:
  DeclVisit(Decl *d, CXCursor parent, bool isFirst) :
    VisitorJob(parent, VisitorJob::DeclVisitKind,
               d, isFirst ? (void*) 1 : (void*) 0) {}
  static bool classof(const VisitorJob *VJ) {
    return VJ->getKind() == DeclVisitKind;
  }
  Decl *get() const { return static_cast<Decl*>(data[0]); }
  bool isFirst() const { return data[1] ? true : false; }
};
class TypeLocVisit : public VisitorJob {
public:
  TypeLocVisit(TypeLoc tl, CXCursor parent) :
    VisitorJob(parent, VisitorJob::TypeLocVisitKind,
               tl.getType().getAsOpaquePtr(), tl.getOpaqueData()) {}

  static bool classof(const VisitorJob *VJ) {
    return VJ->getKind() == TypeLocVisitKind;
  }

  TypeLoc get() const { 
    QualType T = QualType::getFromOpaquePtr(data[0]);
    return TypeLoc(T, data[1]);
  }
};

class LabelRefVisit : public VisitorJob {
public:
  LabelRefVisit(LabelDecl *LD, SourceLocation labelLoc, CXCursor parent)
    : VisitorJob(parent, VisitorJob::LabelRefVisitKind, LD,
                 labelLoc.getPtrEncoding()) {}
  
  static bool classof(const VisitorJob *VJ) {
    return VJ->getKind() == VisitorJob::LabelRefVisitKind;
  }
  LabelDecl *get() const { return static_cast<LabelDecl*>(data[0]); }
  SourceLocation getLoc() const { 
    return SourceLocation::getFromPtrEncoding(data[1]); }
};
class NestedNameSpecifierVisit : public VisitorJob {
public:
  NestedNameSpecifierVisit(NestedNameSpecifier *NS, SourceRange R,
                           CXCursor parent)
    : VisitorJob(parent, VisitorJob::NestedNameSpecifierVisitKind,
                 NS, R.getBegin().getPtrEncoding(),
                 R.getEnd().getPtrEncoding()) {}
  static bool classof(const VisitorJob *VJ) {
    return VJ->getKind() == VisitorJob::NestedNameSpecifierVisitKind;
  }
  NestedNameSpecifier *get() const {
    return static_cast<NestedNameSpecifier*>(data[0]);
  }
  SourceRange getSourceRange() const {
    SourceLocation A =
      SourceLocation::getFromRawEncoding((unsigned)(uintptr_t) data[1]);
    SourceLocation B =
      SourceLocation::getFromRawEncoding((unsigned)(uintptr_t) data[2]);
    return SourceRange(A, B);
  }
};
  
class NestedNameSpecifierLocVisit : public VisitorJob {
public:
  NestedNameSpecifierLocVisit(NestedNameSpecifierLoc Qualifier, CXCursor parent)
    : VisitorJob(parent, VisitorJob::NestedNameSpecifierLocVisitKind,
                 Qualifier.getNestedNameSpecifier(),
                 Qualifier.getOpaqueData()) { }
  
  static bool classof(const VisitorJob *VJ) {
    return VJ->getKind() == VisitorJob::NestedNameSpecifierLocVisitKind;
  }
  
  NestedNameSpecifierLoc get() const {
    return NestedNameSpecifierLoc(static_cast<NestedNameSpecifier*>(data[0]), 
                                  data[1]);
  }
};
  
class DeclarationNameInfoVisit : public VisitorJob {
public:
  DeclarationNameInfoVisit(Stmt *S, CXCursor parent)
    : VisitorJob(parent, VisitorJob::DeclarationNameInfoVisitKind, S) {}
  static bool classof(const VisitorJob *VJ) {
    return VJ->getKind() == VisitorJob::DeclarationNameInfoVisitKind;
  }
  DeclarationNameInfo get() const {
    Stmt *S = static_cast<Stmt*>(data[0]);
    switch (S->getStmtClass()) {
    default:
      llvm_unreachable("Unhandled Stmt");
    case Stmt::CXXDependentScopeMemberExprClass:
      return cast<CXXDependentScopeMemberExpr>(S)->getMemberNameInfo();
    case Stmt::DependentScopeDeclRefExprClass:
      return cast<DependentScopeDeclRefExpr>(S)->getNameInfo();
    }
  }
};
class MemberRefVisit : public VisitorJob {
public:
  MemberRefVisit(FieldDecl *D, SourceLocation L, CXCursor parent)
    : VisitorJob(parent, VisitorJob::MemberRefVisitKind, D,
                 L.getPtrEncoding()) {}
  static bool classof(const VisitorJob *VJ) {
    return VJ->getKind() == VisitorJob::MemberRefVisitKind;
  }
  FieldDecl *get() const {
    return static_cast<FieldDecl*>(data[0]);
  }
  SourceLocation getLoc() const {
    return SourceLocation::getFromRawEncoding((unsigned)(uintptr_t) data[1]);
  }
};
class EnqueueVisitor : public StmtVisitor<EnqueueVisitor, void> {
  VisitorWorkList &WL;
  CXCursor Parent;
public:
  EnqueueVisitor(VisitorWorkList &wl, CXCursor parent)
    : WL(wl), Parent(parent) {}

  void VisitAddrLabelExpr(AddrLabelExpr *E);
  void VisitBlockExpr(BlockExpr *B);
  void VisitCompoundLiteralExpr(CompoundLiteralExpr *E);
  void VisitCompoundStmt(CompoundStmt *S);
  void VisitCXXDefaultArgExpr(CXXDefaultArgExpr *E) { /* Do nothing. */ }
  void VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr *E);
  void VisitCXXNewExpr(CXXNewExpr *E);
  void VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E);
  void VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E);
  void VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E);
  void VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *E);
  void VisitCXXTypeidExpr(CXXTypeidExpr *E);
  void VisitCXXUnresolvedConstructExpr(CXXUnresolvedConstructExpr *E);
  void VisitCXXUuidofExpr(CXXUuidofExpr *E);
  void VisitDeclRefExpr(DeclRefExpr *D);
  void VisitDeclStmt(DeclStmt *S);
  void VisitDependentScopeDeclRefExpr(DependentScopeDeclRefExpr *E);
  void VisitDesignatedInitExpr(DesignatedInitExpr *E);
  void VisitExplicitCastExpr(ExplicitCastExpr *E);
  void VisitForStmt(ForStmt *FS);
  void VisitGotoStmt(GotoStmt *GS);
  void VisitIfStmt(IfStmt *If);
  void VisitInitListExpr(InitListExpr *IE);
  void VisitMemberExpr(MemberExpr *M);
  void VisitOffsetOfExpr(OffsetOfExpr *E);
  void VisitObjCEncodeExpr(ObjCEncodeExpr *E);
  void VisitObjCMessageExpr(ObjCMessageExpr *M);
  void VisitOverloadExpr(OverloadExpr *E);
  void VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E);
  void VisitStmt(Stmt *S);
  void VisitSwitchStmt(SwitchStmt *S);
  void VisitWhileStmt(WhileStmt *W);
  void VisitUnaryTypeTraitExpr(UnaryTypeTraitExpr *E);
  void VisitBinaryTypeTraitExpr(BinaryTypeTraitExpr *E);
  void VisitUnresolvedMemberExpr(UnresolvedMemberExpr *U);
  void VisitVAArgExpr(VAArgExpr *E);
  void VisitSizeOfPackExpr(SizeOfPackExpr *E);
  
private:
  void AddDeclarationNameInfo(Stmt *S);
  void AddNestedNameSpecifier(NestedNameSpecifier *NS, SourceRange R);
  void AddNestedNameSpecifierLoc(NestedNameSpecifierLoc Qualifier);
  void AddExplicitTemplateArgs(const ExplicitTemplateArgumentList *A);
  void AddMemberRef(FieldDecl *D, SourceLocation L);
  void AddStmt(Stmt *S);
  void AddDecl(Decl *D, bool isFirst = true);
  void AddTypeLoc(TypeSourceInfo *TI);
  void EnqueueChildren(Stmt *S);
};
} // end anonyous namespace

void EnqueueVisitor::AddDeclarationNameInfo(Stmt *S) {
  // 'S' should always be non-null, since it comes from the
  // statement we are visiting.
  WL.push_back(DeclarationNameInfoVisit(S, Parent));
}
void EnqueueVisitor::AddNestedNameSpecifier(NestedNameSpecifier *N,
                                            SourceRange R) {
  if (N)
    WL.push_back(NestedNameSpecifierVisit(N, R, Parent));
}

void 
EnqueueVisitor::AddNestedNameSpecifierLoc(NestedNameSpecifierLoc Qualifier) {
  if (Qualifier)
    WL.push_back(NestedNameSpecifierLocVisit(Qualifier, Parent));
}

void EnqueueVisitor::AddStmt(Stmt *S) {
  if (S)
    WL.push_back(StmtVisit(S, Parent));
}
void EnqueueVisitor::AddDecl(Decl *D, bool isFirst) {
  if (D)
    WL.push_back(DeclVisit(D, Parent, isFirst));
}
void EnqueueVisitor::
  AddExplicitTemplateArgs(const ExplicitTemplateArgumentList *A) {
  if (A)
    WL.push_back(ExplicitTemplateArgsVisit(
                        const_cast<ExplicitTemplateArgumentList*>(A), Parent));
}
void EnqueueVisitor::AddMemberRef(FieldDecl *D, SourceLocation L) {
  if (D)
    WL.push_back(MemberRefVisit(D, L, Parent));
}
void EnqueueVisitor::AddTypeLoc(TypeSourceInfo *TI) {
  if (TI)
    WL.push_back(TypeLocVisit(TI->getTypeLoc(), Parent));
 }
void EnqueueVisitor::EnqueueChildren(Stmt *S) {
  unsigned size = WL.size();
  for (Stmt::child_range Child = S->children(); Child; ++Child) {
    AddStmt(*Child);
  }
  if (size == WL.size())
    return;
  // Now reverse the entries we just added.  This will match the DFS
  // ordering performed by the worklist.
  VisitorWorkList::iterator I = WL.begin() + size, E = WL.end();
  std::reverse(I, E);
}
void EnqueueVisitor::VisitAddrLabelExpr(AddrLabelExpr *E) {
  WL.push_back(LabelRefVisit(E->getLabel(), E->getLabelLoc(), Parent));
}
void EnqueueVisitor::VisitBlockExpr(BlockExpr *B) {
  AddDecl(B->getBlockDecl());
}
void EnqueueVisitor::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  EnqueueChildren(E);
  AddTypeLoc(E->getTypeSourceInfo());
}
void EnqueueVisitor::VisitCompoundStmt(CompoundStmt *S) {
  for (CompoundStmt::reverse_body_iterator I = S->body_rbegin(),
        E = S->body_rend(); I != E; ++I) {
    AddStmt(*I);
  }
}
void EnqueueVisitor::
VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr *E) {
  AddExplicitTemplateArgs(E->getOptionalExplicitTemplateArgs());
  AddDeclarationNameInfo(E);
  if (NestedNameSpecifierLoc QualifierLoc = E->getQualifierLoc())
    AddNestedNameSpecifierLoc(QualifierLoc);
  if (!E->isImplicitAccess())
    AddStmt(E->getBase());
}
void EnqueueVisitor::VisitCXXNewExpr(CXXNewExpr *E) {
  // Enqueue the initializer or constructor arguments.
  for (unsigned I = E->getNumConstructorArgs(); I > 0; --I)
    AddStmt(E->getConstructorArg(I-1));
  // Enqueue the array size, if any.
  AddStmt(E->getArraySize());
  // Enqueue the allocated type.
  AddTypeLoc(E->getAllocatedTypeSourceInfo());
  // Enqueue the placement arguments.
  for (unsigned I = E->getNumPlacementArgs(); I > 0; --I)
    AddStmt(E->getPlacementArg(I-1));
}
void EnqueueVisitor::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *CE) {
  for (unsigned I = CE->getNumArgs(); I > 1 /* Yes, this is 1 */; --I)
    AddStmt(CE->getArg(I-1));
  AddStmt(CE->getCallee());
  AddStmt(CE->getArg(0));
}
void EnqueueVisitor::VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E) {
  // Visit the name of the type being destroyed.
  AddTypeLoc(E->getDestroyedTypeInfo());
  // Visit the scope type that looks disturbingly like the nested-name-specifier
  // but isn't.
  AddTypeLoc(E->getScopeTypeInfo());
  // Visit the nested-name-specifier.
  if (NestedNameSpecifierLoc QualifierLoc = E->getQualifierLoc())
    AddNestedNameSpecifierLoc(QualifierLoc);
  // Visit base expression.
  AddStmt(E->getBase());
}
void EnqueueVisitor::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E) {
  AddTypeLoc(E->getTypeSourceInfo());
}
void EnqueueVisitor::VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *E) {
  EnqueueChildren(E);
  AddTypeLoc(E->getTypeSourceInfo());
}
void EnqueueVisitor::VisitCXXTypeidExpr(CXXTypeidExpr *E) {
  EnqueueChildren(E);
  if (E->isTypeOperand())
    AddTypeLoc(E->getTypeOperandSourceInfo());
}

void EnqueueVisitor::VisitCXXUnresolvedConstructExpr(CXXUnresolvedConstructExpr 
                                                     *E) {
  EnqueueChildren(E);
  AddTypeLoc(E->getTypeSourceInfo());
}
void EnqueueVisitor::VisitCXXUuidofExpr(CXXUuidofExpr *E) {
  EnqueueChildren(E);
  if (E->isTypeOperand())
    AddTypeLoc(E->getTypeOperandSourceInfo());
}
void EnqueueVisitor::VisitDeclRefExpr(DeclRefExpr *DR) {
  if (DR->hasExplicitTemplateArgs()) {
    AddExplicitTemplateArgs(&DR->getExplicitTemplateArgs());
  }
  WL.push_back(DeclRefExprParts(DR, Parent));
}
void EnqueueVisitor::VisitDependentScopeDeclRefExpr(DependentScopeDeclRefExpr *E) {
  AddExplicitTemplateArgs(E->getOptionalExplicitTemplateArgs());
  AddDeclarationNameInfo(E);
  AddNestedNameSpecifierLoc(E->getQualifierLoc());
}
void EnqueueVisitor::VisitDeclStmt(DeclStmt *S) {
  unsigned size = WL.size();
  bool isFirst = true;
  for (DeclStmt::decl_iterator D = S->decl_begin(), DEnd = S->decl_end();
       D != DEnd; ++D) {
    AddDecl(*D, isFirst);
    isFirst = false;
  }
  if (size == WL.size())
    return;
  // Now reverse the entries we just added.  This will match the DFS
  // ordering performed by the worklist.
  VisitorWorkList::iterator I = WL.begin() + size, E = WL.end();
  std::reverse(I, E);
}
void EnqueueVisitor::VisitDesignatedInitExpr(DesignatedInitExpr *E) {
  AddStmt(E->getInit());
  typedef DesignatedInitExpr::Designator Designator;
  for (DesignatedInitExpr::reverse_designators_iterator
         D = E->designators_rbegin(), DEnd = E->designators_rend();
         D != DEnd; ++D) {
    if (D->isFieldDesignator()) {
      if (FieldDecl *Field = D->getField())
        AddMemberRef(Field, D->getFieldLoc());
      continue;
    }
    if (D->isArrayDesignator()) {
      AddStmt(E->getArrayIndex(*D));
      continue;
    }
    assert(D->isArrayRangeDesignator() && "Unknown designator kind");
    AddStmt(E->getArrayRangeEnd(*D));
    AddStmt(E->getArrayRangeStart(*D));
  }
}
void EnqueueVisitor::VisitExplicitCastExpr(ExplicitCastExpr *E) {
  EnqueueChildren(E);
  AddTypeLoc(E->getTypeInfoAsWritten());
}
void EnqueueVisitor::VisitForStmt(ForStmt *FS) {
  AddStmt(FS->getBody());
  AddStmt(FS->getInc());
  AddStmt(FS->getCond());
  AddDecl(FS->getConditionVariable());
  AddStmt(FS->getInit());
}
void EnqueueVisitor::VisitGotoStmt(GotoStmt *GS) {
  WL.push_back(LabelRefVisit(GS->getLabel(), GS->getLabelLoc(), Parent));
}
void EnqueueVisitor::VisitIfStmt(IfStmt *If) {
  AddStmt(If->getElse());
  AddStmt(If->getThen());
  AddStmt(If->getCond());
  AddDecl(If->getConditionVariable());
}
void EnqueueVisitor::VisitInitListExpr(InitListExpr *IE) {
  // We care about the syntactic form of the initializer list, only.
  if (InitListExpr *Syntactic = IE->getSyntacticForm())
    IE = Syntactic;
  EnqueueChildren(IE);
}
void EnqueueVisitor::VisitMemberExpr(MemberExpr *M) {
  WL.push_back(MemberExprParts(M, Parent));
  
  // If the base of the member access expression is an implicit 'this', don't
  // visit it.
  // FIXME: If we ever want to show these implicit accesses, this will be
  // unfortunate. However, clang_getCursor() relies on this behavior.
  if (!M->isImplicitAccess())
    AddStmt(M->getBase());
}
void EnqueueVisitor::VisitObjCEncodeExpr(ObjCEncodeExpr *E) {
  AddTypeLoc(E->getEncodedTypeSourceInfo());
}
void EnqueueVisitor::VisitObjCMessageExpr(ObjCMessageExpr *M) {
  EnqueueChildren(M);
  AddTypeLoc(M->getClassReceiverTypeInfo());
}
void EnqueueVisitor::VisitOffsetOfExpr(OffsetOfExpr *E) {
  // Visit the components of the offsetof expression.
  for (unsigned N = E->getNumComponents(), I = N; I > 0; --I) {
    typedef OffsetOfExpr::OffsetOfNode OffsetOfNode;
    const OffsetOfNode &Node = E->getComponent(I-1);
    switch (Node.getKind()) {
    case OffsetOfNode::Array:
      AddStmt(E->getIndexExpr(Node.getArrayExprIndex()));
      break;
    case OffsetOfNode::Field:
      AddMemberRef(Node.getField(), Node.getRange().getEnd());
      break;
    case OffsetOfNode::Identifier:
    case OffsetOfNode::Base:
      continue;
    }
  }
  // Visit the type into which we're computing the offset.
  AddTypeLoc(E->getTypeSourceInfo());
}
void EnqueueVisitor::VisitOverloadExpr(OverloadExpr *E) {
  AddExplicitTemplateArgs(E->getOptionalExplicitTemplateArgs());
  WL.push_back(OverloadExprParts(E, Parent));
}
void EnqueueVisitor::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E) {
  EnqueueChildren(E);
  if (E->isArgumentType())
    AddTypeLoc(E->getArgumentTypeInfo());
}
void EnqueueVisitor::VisitStmt(Stmt *S) {
  EnqueueChildren(S);
}
void EnqueueVisitor::VisitSwitchStmt(SwitchStmt *S) {
  AddStmt(S->getBody());
  AddStmt(S->getCond());
  AddDecl(S->getConditionVariable());
}

void EnqueueVisitor::VisitWhileStmt(WhileStmt *W) {
  AddStmt(W->getBody());
  AddStmt(W->getCond());
  AddDecl(W->getConditionVariable());
}
void EnqueueVisitor::VisitUnaryTypeTraitExpr(UnaryTypeTraitExpr *E) {
  AddTypeLoc(E->getQueriedTypeSourceInfo());
}

void EnqueueVisitor::VisitBinaryTypeTraitExpr(BinaryTypeTraitExpr *E) {
  AddTypeLoc(E->getRhsTypeSourceInfo());
  AddTypeLoc(E->getLhsTypeSourceInfo());
}

void EnqueueVisitor::VisitUnresolvedMemberExpr(UnresolvedMemberExpr *U) {
  VisitOverloadExpr(U);
  if (!U->isImplicitAccess())
    AddStmt(U->getBase());
}
void EnqueueVisitor::VisitVAArgExpr(VAArgExpr *E) {
  AddStmt(E->getSubExpr());
  AddTypeLoc(E->getWrittenTypeInfo());
}
void EnqueueVisitor::VisitSizeOfPackExpr(SizeOfPackExpr *E) {
  WL.push_back(SizeOfPackExprParts(E, Parent));
}

void CursorVisitor::EnqueueWorkList(VisitorWorkList &WL, Stmt *S) {
  EnqueueVisitor(WL, MakeCXCursor(S, StmtParent, TU)).Visit(S);
}

bool CursorVisitor::IsInRegionOfInterest(CXCursor C) {
  if (RegionOfInterest.isValid()) {
    SourceRange Range = getRawCursorExtent(C);
    if (Range.isInvalid() || CompareRegionOfInterest(Range))
      return false;
  }
  return true;
}

bool CursorVisitor::RunVisitorWorkList(VisitorWorkList &WL) {
  while (!WL.empty()) {
    // Dequeue the worklist item.
    VisitorJob LI = WL.back();
    WL.pop_back();

    // Set the Parent field, then back to its old value once we're done.
    SetParentRAII SetParent(Parent, StmtParent, LI.getParent());
  
    switch (LI.getKind()) {
      case VisitorJob::DeclVisitKind: {
        Decl *D = cast<DeclVisit>(&LI)->get();
        if (!D)
          continue;

        // For now, perform default visitation for Decls.
        if (Visit(MakeCXCursor(D, TU, cast<DeclVisit>(&LI)->isFirst())))
            return true;

        continue;
      }
      case VisitorJob::ExplicitTemplateArgsVisitKind: {
        const ExplicitTemplateArgumentList *ArgList =
          cast<ExplicitTemplateArgsVisit>(&LI)->get();
        for (const TemplateArgumentLoc *Arg = ArgList->getTemplateArgs(),
               *ArgEnd = Arg + ArgList->NumTemplateArgs;
               Arg != ArgEnd; ++Arg) {
          if (VisitTemplateArgumentLoc(*Arg))
            return true;
        }
        continue;
      }
      case VisitorJob::TypeLocVisitKind: {
        // Perform default visitation for TypeLocs.
        if (Visit(cast<TypeLocVisit>(&LI)->get()))
          return true;
        continue;
      }
      case VisitorJob::LabelRefVisitKind: {
        LabelDecl *LS = cast<LabelRefVisit>(&LI)->get();
        if (LabelStmt *stmt = LS->getStmt()) {
          if (Visit(MakeCursorLabelRef(stmt, cast<LabelRefVisit>(&LI)->getLoc(),
                                       TU))) {
            return true;
          }
        }
        continue;
      }
        
      case VisitorJob::NestedNameSpecifierVisitKind: {
        NestedNameSpecifierVisit *V = cast<NestedNameSpecifierVisit>(&LI);
        if (VisitNestedNameSpecifier(V->get(), V->getSourceRange()))
          return true;
        continue;
      }
        
      case VisitorJob::NestedNameSpecifierLocVisitKind: {
        NestedNameSpecifierLocVisit *V = cast<NestedNameSpecifierLocVisit>(&LI);
        if (VisitNestedNameSpecifierLoc(V->get()))
          return true;
        continue;
      }
        
      case VisitorJob::DeclarationNameInfoVisitKind: {
        if (VisitDeclarationNameInfo(cast<DeclarationNameInfoVisit>(&LI)
                                     ->get()))
          return true;
        continue;
      }
      case VisitorJob::MemberRefVisitKind: {
        MemberRefVisit *V = cast<MemberRefVisit>(&LI);
        if (Visit(MakeCursorMemberRef(V->get(), V->getLoc(), TU)))
          return true;
        continue;
      }
      case VisitorJob::StmtVisitKind: {
        Stmt *S = cast<StmtVisit>(&LI)->get();
        if (!S)
          continue;

        // Update the current cursor.
        CXCursor Cursor = MakeCXCursor(S, StmtParent, TU);
        if (!IsInRegionOfInterest(Cursor))
          continue;
        switch (Visitor(Cursor, Parent, ClientData)) {
          case CXChildVisit_Break: return true;
          case CXChildVisit_Continue: break;
          case CXChildVisit_Recurse:
            EnqueueWorkList(WL, S);
            break;
        }
        continue;
      }
      case VisitorJob::MemberExprPartsKind: {
        // Handle the other pieces in the MemberExpr besides the base.
        MemberExpr *M = cast<MemberExprParts>(&LI)->get();
        
        // Visit the nested-name-specifier
        if (NestedNameSpecifierLoc QualifierLoc = M->getQualifierLoc())
          if (VisitNestedNameSpecifierLoc(QualifierLoc))
            return true;
        
        // Visit the declaration name.
        if (VisitDeclarationNameInfo(M->getMemberNameInfo()))
          return true;
        
        // Visit the explicitly-specified template arguments, if any.
        if (M->hasExplicitTemplateArgs()) {
          for (const TemplateArgumentLoc *Arg = M->getTemplateArgs(),
               *ArgEnd = Arg + M->getNumTemplateArgs();
               Arg != ArgEnd; ++Arg) {
            if (VisitTemplateArgumentLoc(*Arg))
              return true;
          }
        }
        continue;
      }
      case VisitorJob::DeclRefExprPartsKind: {
        DeclRefExpr *DR = cast<DeclRefExprParts>(&LI)->get();
        // Visit nested-name-specifier, if present.
        if (NestedNameSpecifierLoc QualifierLoc = DR->getQualifierLoc())
          if (VisitNestedNameSpecifierLoc(QualifierLoc))
            return true;
        // Visit declaration name.
        if (VisitDeclarationNameInfo(DR->getNameInfo()))
          return true;
        continue;
      }
      case VisitorJob::OverloadExprPartsKind: {
        OverloadExpr *O = cast<OverloadExprParts>(&LI)->get();
        // Visit the nested-name-specifier.
        if (NestedNameSpecifierLoc QualifierLoc = O->getQualifierLoc())
          if (VisitNestedNameSpecifierLoc(QualifierLoc))
            return true;
        // Visit the declaration name.
        if (VisitDeclarationNameInfo(O->getNameInfo()))
          return true;
        // Visit the overloaded declaration reference.
        if (Visit(MakeCursorOverloadedDeclRef(O, TU)))
          return true;
        continue;
      }
      case VisitorJob::SizeOfPackExprPartsKind: {
        SizeOfPackExpr *E = cast<SizeOfPackExprParts>(&LI)->get();
        NamedDecl *Pack = E->getPack();
        if (isa<TemplateTypeParmDecl>(Pack)) {
          if (Visit(MakeCursorTypeRef(cast<TemplateTypeParmDecl>(Pack),
                                      E->getPackLoc(), TU)))
            return true;
          
          continue;
        }
          
        if (isa<TemplateTemplateParmDecl>(Pack)) {
          if (Visit(MakeCursorTemplateRef(cast<TemplateTemplateParmDecl>(Pack),
                                          E->getPackLoc(), TU)))
            return true;
          
          continue;
        }
        
        // Non-type template parameter packs and function parameter packs are
        // treated like DeclRefExpr cursors.
        continue;
      }
    }
  }
  return false;
}

bool CursorVisitor::Visit(Stmt *S) {
  VisitorWorkList *WL = 0;
  if (!WorkListFreeList.empty()) {
    WL = WorkListFreeList.back();
    WL->clear();
    WorkListFreeList.pop_back();
  }
  else {
    WL = new VisitorWorkList();
    WorkListCache.push_back(WL);
  }
  EnqueueWorkList(*WL, S);
  bool result = RunVisitorWorkList(*WL);
  WorkListFreeList.push_back(WL);
  return result;
}

//===----------------------------------------------------------------------===//
// Misc. API hooks.
//===----------------------------------------------------------------------===//               

static llvm::sys::Mutex EnableMultithreadingMutex;
static bool EnabledMultithreading;

extern "C" {
CXIndex clang_createIndex(int excludeDeclarationsFromPCH,
                          int displayDiagnostics) {
  // Disable pretty stack trace functionality, which will otherwise be a very
  // poor citizen of the world and set up all sorts of signal handlers.
  llvm::DisablePrettyStackTrace = true;

  // We use crash recovery to make some of our APIs more reliable, implicitly
  // enable it.
  llvm::CrashRecoveryContext::Enable();

  // Enable support for multithreading in LLVM.
  {
    llvm::sys::ScopedLock L(EnableMultithreadingMutex);
    if (!EnabledMultithreading) {
      llvm::llvm_start_multithreaded();
      EnabledMultithreading = true;
    }
  }

  CIndexer *CIdxr = new CIndexer();
  if (excludeDeclarationsFromPCH)
    CIdxr->setOnlyLocalDecls();
  if (displayDiagnostics)
    CIdxr->setDisplayDiagnostics();
  return CIdxr;
}

void clang_disposeIndex(CXIndex CIdx) {
  if (CIdx)
    delete static_cast<CIndexer *>(CIdx);
}

CXTranslationUnit clang_createTranslationUnit(CXIndex CIdx,
                                              const char *ast_filename) {
  if (!CIdx)
    return 0;

  CIndexer *CXXIdx = static_cast<CIndexer *>(CIdx);
  FileSystemOptions FileSystemOpts;
  FileSystemOpts.WorkingDir = CXXIdx->getWorkingDirectory();

  llvm::IntrusiveRefCntPtr<Diagnostic> Diags;
  ASTUnit *TU = ASTUnit::LoadFromASTFile(ast_filename, Diags, FileSystemOpts,
                                  CXXIdx->getOnlyLocalDecls(),
                                  0, 0, true);
  return MakeCXTranslationUnit(TU);
}

unsigned clang_defaultEditingTranslationUnitOptions() {
  return CXTranslationUnit_PrecompiledPreamble | 
         CXTranslationUnit_CacheCompletionResults |
         CXTranslationUnit_CXXPrecompiledPreamble;
}
  
CXTranslationUnit
clang_createTranslationUnitFromSourceFile(CXIndex CIdx,
                                          const char *source_filename,
                                          int num_command_line_args,
                                          const char * const *command_line_args,
                                          unsigned num_unsaved_files,
                                          struct CXUnsavedFile *unsaved_files) {
  return clang_parseTranslationUnit(CIdx, source_filename,
                                    command_line_args, num_command_line_args,
                                    unsaved_files, num_unsaved_files,
                                 CXTranslationUnit_DetailedPreprocessingRecord);
}
  
struct ParseTranslationUnitInfo {
  CXIndex CIdx;
  const char *source_filename;
  const char *const *command_line_args;
  int num_command_line_args;
  struct CXUnsavedFile *unsaved_files;
  unsigned num_unsaved_files;
  unsigned options;
  CXTranslationUnit result;
};
static void clang_parseTranslationUnit_Impl(void *UserData) {
  ParseTranslationUnitInfo *PTUI =
    static_cast<ParseTranslationUnitInfo*>(UserData);
  CXIndex CIdx = PTUI->CIdx;
  const char *source_filename = PTUI->source_filename;
  const char * const *command_line_args = PTUI->command_line_args;
  int num_command_line_args = PTUI->num_command_line_args;
  struct CXUnsavedFile *unsaved_files = PTUI->unsaved_files;
  unsigned num_unsaved_files = PTUI->num_unsaved_files;
  unsigned options = PTUI->options;
  PTUI->result = 0;

  if (!CIdx)
    return;

  CIndexer *CXXIdx = static_cast<CIndexer *>(CIdx);

  bool PrecompilePreamble = options & CXTranslationUnit_PrecompiledPreamble;
  bool CompleteTranslationUnit
    = ((options & CXTranslationUnit_Incomplete) == 0);
  bool CacheCodeCompetionResults
    = options & CXTranslationUnit_CacheCompletionResults;
  bool CXXPrecompilePreamble
    = options & CXTranslationUnit_CXXPrecompiledPreamble;
  bool CXXChainedPCH
    = options & CXTranslationUnit_CXXChainedPCH;
  
  // Configure the diagnostics.
  DiagnosticOptions DiagOpts;
  llvm::IntrusiveRefCntPtr<Diagnostic> Diags;
  Diags = CompilerInstance::createDiagnostics(DiagOpts, num_command_line_args, 
                                              command_line_args);

  llvm::SmallVector<ASTUnit::RemappedFile, 4> RemappedFiles;
  for (unsigned I = 0; I != num_unsaved_files; ++I) {
    llvm::StringRef Data(unsaved_files[I].Contents, unsaved_files[I].Length);
    const llvm::MemoryBuffer *Buffer
      = llvm::MemoryBuffer::getMemBufferCopy(Data, unsaved_files[I].Filename);
    RemappedFiles.push_back(std::make_pair(unsaved_files[I].Filename,
                                           Buffer));
  }

  llvm::SmallVector<const char *, 16> Args;
  
  // Since the Clang C library is primarily used by batch tools dealing with
  // (often very broken) source code, where spell-checking can have a
  // significant negative impact on performance (particularly when 
  // precompiled headers are involved), we disable it by default.
  // Only do this if we haven't found a spell-checking-related argument.
  bool FoundSpellCheckingArgument = false;
  for (int I = 0; I != num_command_line_args; ++I) {
    if (strcmp(command_line_args[I], "-fno-spell-checking") == 0 ||
        strcmp(command_line_args[I], "-fspell-checking") == 0) {
      FoundSpellCheckingArgument = true;
      break;
    }
  }
  if (!FoundSpellCheckingArgument)
    Args.push_back("-fno-spell-checking");
  
  Args.insert(Args.end(), command_line_args,
              command_line_args + num_command_line_args);

  // The 'source_filename' argument is optional.  If the caller does not
  // specify it then it is assumed that the source file is specified
  // in the actual argument list.
  // Put the source file after command_line_args otherwise if '-x' flag is
  // present it will be unused.
  if (source_filename)
    Args.push_back(source_filename);

  // Do we need the detailed preprocessing record?
  if (options & CXTranslationUnit_DetailedPreprocessingRecord) {
    Args.push_back("-Xclang");
    Args.push_back("-detailed-preprocessing-record");
  }
  
  unsigned NumErrors = Diags->getClient()->getNumErrors();
  llvm::OwningPtr<ASTUnit> Unit(
    ASTUnit::LoadFromCommandLine(Args.data(), Args.data() + Args.size(),
                                 Diags,
                                 CXXIdx->getClangResourcesPath(),
                                 CXXIdx->getOnlyLocalDecls(),
                                 /*CaptureDiagnostics=*/true,
                                 RemappedFiles.data(),
                                 RemappedFiles.size(),
                                 PrecompilePreamble,
                                 CompleteTranslationUnit,
                                 CacheCodeCompetionResults,
                                 CXXPrecompilePreamble,
                                 CXXChainedPCH));

  if (NumErrors != Diags->getClient()->getNumErrors()) {
    // Make sure to check that 'Unit' is non-NULL.
    if (CXXIdx->getDisplayDiagnostics() && Unit.get()) {
      for (ASTUnit::stored_diag_iterator D = Unit->stored_diag_begin(), 
                                      DEnd = Unit->stored_diag_end();
           D != DEnd; ++D) {
        CXStoredDiagnostic Diag(*D, Unit->getASTContext().getLangOptions());
        CXString Msg = clang_formatDiagnostic(&Diag,
                                    clang_defaultDiagnosticDisplayOptions());
        fprintf(stderr, "%s\n", clang_getCString(Msg));
        clang_disposeString(Msg);
      }
#ifdef LLVM_ON_WIN32
      // On Windows, force a flush, since there may be multiple copies of
      // stderr and stdout in the file system, all with different buffers
      // but writing to the same device.
      fflush(stderr);
#endif
    }
  }

  PTUI->result = MakeCXTranslationUnit(Unit.take());
}
CXTranslationUnit clang_parseTranslationUnit(CXIndex CIdx,
                                             const char *source_filename,
                                         const char * const *command_line_args,
                                             int num_command_line_args,
                                            struct CXUnsavedFile *unsaved_files,
                                             unsigned num_unsaved_files,
                                             unsigned options) {
  ParseTranslationUnitInfo PTUI = { CIdx, source_filename, command_line_args,
                                    num_command_line_args, unsaved_files,
                                    num_unsaved_files, options, 0 };
  llvm::CrashRecoveryContext CRC;

  if (!RunSafely(CRC, clang_parseTranslationUnit_Impl, &PTUI)) {
    fprintf(stderr, "libclang: crash detected during parsing: {\n");
    fprintf(stderr, "  'source_filename' : '%s'\n", source_filename);
    fprintf(stderr, "  'command_line_args' : [");
    for (int i = 0; i != num_command_line_args; ++i) {
      if (i)
        fprintf(stderr, ", ");
      fprintf(stderr, "'%s'", command_line_args[i]);
    }
    fprintf(stderr, "],\n");
    fprintf(stderr, "  'unsaved_files' : [");
    for (unsigned i = 0; i != num_unsaved_files; ++i) {
      if (i)
        fprintf(stderr, ", ");
      fprintf(stderr, "('%s', '...', %ld)", unsaved_files[i].Filename,
              unsaved_files[i].Length);
    }
    fprintf(stderr, "],\n");
    fprintf(stderr, "  'options' : %d,\n", options);
    fprintf(stderr, "}\n");
    
    return 0;
  }

  return PTUI.result;
}

unsigned clang_defaultSaveOptions(CXTranslationUnit TU) {
  return CXSaveTranslationUnit_None;
}  
  
int clang_saveTranslationUnit(CXTranslationUnit TU, const char *FileName,
                              unsigned options) {
  if (!TU)
    return 1;
  
  return static_cast<ASTUnit *>(TU->TUData)->Save(FileName);
}

void clang_disposeTranslationUnit(CXTranslationUnit CTUnit) {
  if (CTUnit) {
    // If the translation unit has been marked as unsafe to free, just discard
    // it.
    if (static_cast<ASTUnit *>(CTUnit->TUData)->isUnsafeToFree())
      return;

    delete static_cast<ASTUnit *>(CTUnit->TUData);
    disposeCXStringPool(CTUnit->StringPool);
    delete CTUnit;
  }
}

unsigned clang_defaultReparseOptions(CXTranslationUnit TU) {
  return CXReparse_None;
}

struct ReparseTranslationUnitInfo {
  CXTranslationUnit TU;
  unsigned num_unsaved_files;
  struct CXUnsavedFile *unsaved_files;
  unsigned options;
  int result;
};

static void clang_reparseTranslationUnit_Impl(void *UserData) {
  ReparseTranslationUnitInfo *RTUI =
    static_cast<ReparseTranslationUnitInfo*>(UserData);
  CXTranslationUnit TU = RTUI->TU;
  unsigned num_unsaved_files = RTUI->num_unsaved_files;
  struct CXUnsavedFile *unsaved_files = RTUI->unsaved_files;
  unsigned options = RTUI->options;
  (void) options;
  RTUI->result = 1;

  if (!TU)
    return;

  ASTUnit *CXXUnit = static_cast<ASTUnit *>(TU->TUData);
  ASTUnit::ConcurrencyCheck Check(*CXXUnit);
  
  llvm::SmallVector<ASTUnit::RemappedFile, 4> RemappedFiles;
  for (unsigned I = 0; I != num_unsaved_files; ++I) {
    llvm::StringRef Data(unsaved_files[I].Contents, unsaved_files[I].Length);
    const llvm::MemoryBuffer *Buffer
      = llvm::MemoryBuffer::getMemBufferCopy(Data, unsaved_files[I].Filename);
    RemappedFiles.push_back(std::make_pair(unsaved_files[I].Filename,
                                           Buffer));
  }
  
  if (!CXXUnit->Reparse(RemappedFiles.data(), RemappedFiles.size()))
    RTUI->result = 0;
}

int clang_reparseTranslationUnit(CXTranslationUnit TU,
                                 unsigned num_unsaved_files,
                                 struct CXUnsavedFile *unsaved_files,
                                 unsigned options) {
  ReparseTranslationUnitInfo RTUI = { TU, num_unsaved_files, unsaved_files,
                                      options, 0 };
  llvm::CrashRecoveryContext CRC;

  if (!RunSafely(CRC, clang_reparseTranslationUnit_Impl, &RTUI)) {
    fprintf(stderr, "libclang: crash detected during reparsing\n");
    static_cast<ASTUnit *>(TU->TUData)->setUnsafeToFree(true);
    return 1;
  }


  return RTUI.result;
}


CXString clang_getTranslationUnitSpelling(CXTranslationUnit CTUnit) {
  if (!CTUnit)
    return createCXString("");

  ASTUnit *CXXUnit = static_cast<ASTUnit *>(CTUnit->TUData);
  return createCXString(CXXUnit->getOriginalSourceFileName(), true);
}

CXCursor clang_getTranslationUnitCursor(CXTranslationUnit TU) {
  CXCursor Result = { CXCursor_TranslationUnit, { 0, 0, TU } };
  return Result;
}

} // end: extern "C"

//===----------------------------------------------------------------------===//
// CXSourceLocation and CXSourceRange Operations.
//===----------------------------------------------------------------------===//

extern "C" {
CXSourceLocation clang_getNullLocation() {
  CXSourceLocation Result = { { 0, 0 }, 0 };
  return Result;
}

unsigned clang_equalLocations(CXSourceLocation loc1, CXSourceLocation loc2) {
  return (loc1.ptr_data[0] == loc2.ptr_data[0] &&
          loc1.ptr_data[1] == loc2.ptr_data[1] &&
          loc1.int_data == loc2.int_data);
}

CXSourceLocation clang_getLocation(CXTranslationUnit tu,
                                   CXFile file,
                                   unsigned line,
                                   unsigned column) {
  if (!tu || !file)
    return clang_getNullLocation();
  
  bool Logging = ::getenv("LIBCLANG_LOGGING");
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(tu->TUData);
  const FileEntry *File = static_cast<const FileEntry *>(file);
  SourceLocation SLoc
    = CXXUnit->getSourceManager().getLocation(File, line, column);
  if (SLoc.isInvalid()) {
    if (Logging)
      llvm::errs() << "clang_getLocation(\"" << File->getName() 
                   << "\", " << line << ", " << column << ") = invalid\n";
    return clang_getNullLocation();
  }

  if (Logging)
    llvm::errs() << "clang_getLocation(\"" << File->getName() 
                 << "\", " << line << ", " << column << ") = " 
                 << SLoc.getRawEncoding() << "\n";

  return cxloc::translateSourceLocation(CXXUnit->getASTContext(), SLoc);
}

CXSourceLocation clang_getLocationForOffset(CXTranslationUnit tu,
                                            CXFile file,
                                            unsigned offset) {
  if (!tu || !file)
    return clang_getNullLocation();
  
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(tu->TUData);
  SourceLocation Start 
    = CXXUnit->getSourceManager().getLocation(
                                        static_cast<const FileEntry *>(file),
                                              1, 1);
  if (Start.isInvalid()) return clang_getNullLocation();

  SourceLocation SLoc = Start.getFileLocWithOffset(offset);

  if (SLoc.isInvalid()) return clang_getNullLocation();

  return cxloc::translateSourceLocation(CXXUnit->getASTContext(), SLoc);
}

CXSourceRange clang_getNullRange() {
  CXSourceRange Result = { { 0, 0 }, 0, 0 };
  return Result;
}

CXSourceRange clang_getRange(CXSourceLocation begin, CXSourceLocation end) {
  if (begin.ptr_data[0] != end.ptr_data[0] ||
      begin.ptr_data[1] != end.ptr_data[1])
    return clang_getNullRange();

  CXSourceRange Result = { { begin.ptr_data[0], begin.ptr_data[1] },
                           begin.int_data, end.int_data };
  return Result;
}

void clang_getInstantiationLocation(CXSourceLocation location,
                                    CXFile *file,
                                    unsigned *line,
                                    unsigned *column,
                                    unsigned *offset) {
  SourceLocation Loc = SourceLocation::getFromRawEncoding(location.int_data);

  if (!location.ptr_data[0] || Loc.isInvalid()) {
    if (file)
      *file = 0;
    if (line)
      *line = 0;
    if (column)
      *column = 0;
    if (offset)
      *offset = 0;
    return;
  }

  const SourceManager &SM =
    *static_cast<const SourceManager*>(location.ptr_data[0]);
  SourceLocation InstLoc = SM.getInstantiationLoc(Loc);

  if (file)
    *file = (void *)SM.getFileEntryForID(SM.getFileID(InstLoc));
  if (line)
    *line = SM.getInstantiationLineNumber(InstLoc);
  if (column)
    *column = SM.getInstantiationColumnNumber(InstLoc);
  if (offset)
    *offset = SM.getDecomposedLoc(InstLoc).second;
}

void clang_getSpellingLocation(CXSourceLocation location,
                               CXFile *file,
                               unsigned *line,
                               unsigned *column,
                               unsigned *offset) {
  SourceLocation Loc = SourceLocation::getFromRawEncoding(location.int_data);

  if (!location.ptr_data[0] || Loc.isInvalid()) {
    if (file)
      *file = 0;
    if (line)
      *line = 0;
    if (column)
      *column = 0;
    if (offset)
      *offset = 0;
    return;
  }

  const SourceManager &SM =
    *static_cast<const SourceManager*>(location.ptr_data[0]);
  SourceLocation SpellLoc = Loc;
  if (SpellLoc.isMacroID()) {
    SourceLocation SimpleSpellingLoc = SM.getImmediateSpellingLoc(SpellLoc);
    if (SimpleSpellingLoc.isFileID() &&
        SM.getFileEntryForID(SM.getDecomposedLoc(SimpleSpellingLoc).first))
      SpellLoc = SimpleSpellingLoc;
    else
      SpellLoc = SM.getInstantiationLoc(SpellLoc);
  }

  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(SpellLoc);
  FileID FID = LocInfo.first;
  unsigned FileOffset = LocInfo.second;

  if (file)
    *file = (void *)SM.getFileEntryForID(FID);
  if (line)
    *line = SM.getLineNumber(FID, FileOffset);
  if (column)
    *column = SM.getColumnNumber(FID, FileOffset);
  if (offset)
    *offset = FileOffset;
}

CXSourceLocation clang_getRangeStart(CXSourceRange range) {
  CXSourceLocation Result = { { range.ptr_data[0], range.ptr_data[1] },
                              range.begin_int_data };
  return Result;
}

CXSourceLocation clang_getRangeEnd(CXSourceRange range) {
  CXSourceLocation Result = { { range.ptr_data[0], range.ptr_data[1] },
                              range.end_int_data };
  return Result;
}

} // end: extern "C"

//===----------------------------------------------------------------------===//
// CXFile Operations.
//===----------------------------------------------------------------------===//

extern "C" {
CXString clang_getFileName(CXFile SFile) {
  if (!SFile)
    return createCXString((const char*)NULL);

  FileEntry *FEnt = static_cast<FileEntry *>(SFile);
  return createCXString(FEnt->getName());
}

time_t clang_getFileTime(CXFile SFile) {
  if (!SFile)
    return 0;

  FileEntry *FEnt = static_cast<FileEntry *>(SFile);
  return FEnt->getModificationTime();
}

CXFile clang_getFile(CXTranslationUnit tu, const char *file_name) {
  if (!tu)
    return 0;

  ASTUnit *CXXUnit = static_cast<ASTUnit *>(tu->TUData);

  FileManager &FMgr = CXXUnit->getFileManager();
  return const_cast<FileEntry *>(FMgr.getFile(file_name));
}

} // end: extern "C"

//===----------------------------------------------------------------------===//
// CXCursor Operations.
//===----------------------------------------------------------------------===//

static Decl *getDeclFromExpr(Stmt *E) {
  if (CastExpr *CE = dyn_cast<CastExpr>(E))
    return getDeclFromExpr(CE->getSubExpr());

  if (DeclRefExpr *RefExpr = dyn_cast<DeclRefExpr>(E))
    return RefExpr->getDecl();
  if (BlockDeclRefExpr *RefExpr = dyn_cast<BlockDeclRefExpr>(E))
    return RefExpr->getDecl();
  if (MemberExpr *ME = dyn_cast<MemberExpr>(E))
    return ME->getMemberDecl();
  if (ObjCIvarRefExpr *RE = dyn_cast<ObjCIvarRefExpr>(E))
    return RE->getDecl();
  if (ObjCPropertyRefExpr *PRE = dyn_cast<ObjCPropertyRefExpr>(E))
    return PRE->isExplicitProperty() ? PRE->getExplicitProperty() : 0;
      
  if (CallExpr *CE = dyn_cast<CallExpr>(E))
    return getDeclFromExpr(CE->getCallee());
  if (CXXConstructExpr *CE = llvm::dyn_cast<CXXConstructExpr>(E))
    if (!CE->isElidable())
    return CE->getConstructor();
  if (ObjCMessageExpr *OME = dyn_cast<ObjCMessageExpr>(E))
    return OME->getMethodDecl();

  if (ObjCProtocolExpr *PE = dyn_cast<ObjCProtocolExpr>(E))
    return PE->getProtocol();
  if (SubstNonTypeTemplateParmPackExpr *NTTP 
                              = dyn_cast<SubstNonTypeTemplateParmPackExpr>(E))
    return NTTP->getParameterPack();
  if (SizeOfPackExpr *SizeOfPack = dyn_cast<SizeOfPackExpr>(E))
    if (isa<NonTypeTemplateParmDecl>(SizeOfPack->getPack()) || 
        isa<ParmVarDecl>(SizeOfPack->getPack()))
      return SizeOfPack->getPack();
  
  return 0;
}

static SourceLocation getLocationFromExpr(Expr *E) {
  if (ObjCMessageExpr *Msg = dyn_cast<ObjCMessageExpr>(E))
    return /*FIXME:*/Msg->getLeftLoc();
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E))
    return DRE->getLocation();
  if (BlockDeclRefExpr *RefExpr = dyn_cast<BlockDeclRefExpr>(E))
    return RefExpr->getLocation();
  if (MemberExpr *Member = dyn_cast<MemberExpr>(E))
    return Member->getMemberLoc();
  if (ObjCIvarRefExpr *Ivar = dyn_cast<ObjCIvarRefExpr>(E))
    return Ivar->getLocation();
  if (SizeOfPackExpr *SizeOfPack = dyn_cast<SizeOfPackExpr>(E))
    return SizeOfPack->getPackLoc();
  
  return E->getLocStart();
}

extern "C" {

unsigned clang_visitChildren(CXCursor parent,
                             CXCursorVisitor visitor,
                             CXClientData client_data) {
  CursorVisitor CursorVis(getCursorTU(parent), visitor, client_data, 
                          getCursorASTUnit(parent)->getMaxPCHLevel());
  return CursorVis.VisitChildren(parent);
}

#ifndef __has_feature
#define __has_feature(x) 0
#endif
#if __has_feature(blocks)
typedef enum CXChildVisitResult 
     (^CXCursorVisitorBlock)(CXCursor cursor, CXCursor parent);

static enum CXChildVisitResult visitWithBlock(CXCursor cursor, CXCursor parent,
    CXClientData client_data) {
  CXCursorVisitorBlock block = (CXCursorVisitorBlock)client_data;
  return block(cursor, parent);
}
#else
// If we are compiled with a compiler that doesn't have native blocks support,
// define and call the block manually, so the 
typedef struct _CXChildVisitResult
{
	void *isa;
	int flags;
	int reserved;
	enum CXChildVisitResult(*invoke)(struct _CXChildVisitResult*, CXCursor,
                                         CXCursor);
} *CXCursorVisitorBlock;

static enum CXChildVisitResult visitWithBlock(CXCursor cursor, CXCursor parent,
    CXClientData client_data) {
  CXCursorVisitorBlock block = (CXCursorVisitorBlock)client_data;
  return block->invoke(block, cursor, parent);
}
#endif


unsigned clang_visitChildrenWithBlock(CXCursor parent,
                                      CXCursorVisitorBlock block) {
  return clang_visitChildren(parent, visitWithBlock, block);
}

static CXString getDeclSpelling(Decl *D) {
  NamedDecl *ND = dyn_cast_or_null<NamedDecl>(D);
  if (!ND) {
    if (ObjCPropertyImplDecl *PropImpl =llvm::dyn_cast<ObjCPropertyImplDecl>(D))
      if (ObjCPropertyDecl *Property = PropImpl->getPropertyDecl())
        return createCXString(Property->getIdentifier()->getName());
    
    return createCXString("");
  }
  
  if (ObjCMethodDecl *OMD = dyn_cast<ObjCMethodDecl>(ND))
    return createCXString(OMD->getSelector().getAsString());

  if (ObjCCategoryImplDecl *CIMP = dyn_cast<ObjCCategoryImplDecl>(ND))
    // No, this isn't the same as the code below. getIdentifier() is non-virtual
    // and returns different names. NamedDecl returns the class name and
    // ObjCCategoryImplDecl returns the category name.
    return createCXString(CIMP->getIdentifier()->getNameStart());

  if (isa<UsingDirectiveDecl>(D))
    return createCXString("");
  
  llvm::SmallString<1024> S;
  llvm::raw_svector_ostream os(S);
  ND->printName(os);
  
  return createCXString(os.str());
}

CXString clang_getCursorSpelling(CXCursor C) {
  if (clang_isTranslationUnit(C.kind))
    return clang_getTranslationUnitSpelling(
                            static_cast<CXTranslationUnit>(C.data[2]));

  if (clang_isReference(C.kind)) {
    switch (C.kind) {
    case CXCursor_ObjCSuperClassRef: {
      ObjCInterfaceDecl *Super = getCursorObjCSuperClassRef(C).first;
      return createCXString(Super->getIdentifier()->getNameStart());
    }
    case CXCursor_ObjCClassRef: {
      ObjCInterfaceDecl *Class = getCursorObjCClassRef(C).first;
      return createCXString(Class->getIdentifier()->getNameStart());
    }
    case CXCursor_ObjCProtocolRef: {
      ObjCProtocolDecl *OID = getCursorObjCProtocolRef(C).first;
      assert(OID && "getCursorSpelling(): Missing protocol decl");
      return createCXString(OID->getIdentifier()->getNameStart());
    }
    case CXCursor_CXXBaseSpecifier: {
      CXXBaseSpecifier *B = getCursorCXXBaseSpecifier(C);
      return createCXString(B->getType().getAsString());
    }
    case CXCursor_TypeRef: {
      TypeDecl *Type = getCursorTypeRef(C).first;
      assert(Type && "Missing type decl");

      return createCXString(getCursorContext(C).getTypeDeclType(Type).
                              getAsString());
    }
    case CXCursor_TemplateRef: {
      TemplateDecl *Template = getCursorTemplateRef(C).first;
      assert(Template && "Missing template decl");
      
      return createCXString(Template->getNameAsString());
    }
        
    case CXCursor_NamespaceRef: {
      NamedDecl *NS = getCursorNamespaceRef(C).first;
      assert(NS && "Missing namespace decl");
      
      return createCXString(NS->getNameAsString());
    }

    case CXCursor_MemberRef: {
      FieldDecl *Field = getCursorMemberRef(C).first;
      assert(Field && "Missing member decl");
      
      return createCXString(Field->getNameAsString());
    }

    case CXCursor_LabelRef: {
      LabelStmt *Label = getCursorLabelRef(C).first;
      assert(Label && "Missing label");
      
      return createCXString(Label->getName());
    }

    case CXCursor_OverloadedDeclRef: {
      OverloadedDeclRefStorage Storage = getCursorOverloadedDeclRef(C).first;
      if (Decl *D = Storage.dyn_cast<Decl *>()) {
        if (NamedDecl *ND = dyn_cast<NamedDecl>(D))
          return createCXString(ND->getNameAsString());
        return createCXString("");
      }
      if (OverloadExpr *E = Storage.dyn_cast<OverloadExpr *>())
        return createCXString(E->getName().getAsString());
      OverloadedTemplateStorage *Ovl
        = Storage.get<OverloadedTemplateStorage*>();
      if (Ovl->size() == 0)
        return createCXString("");
      return createCXString((*Ovl->begin())->getNameAsString());
    }
        
    default:
      return createCXString("<not implemented>");
    }
  }

  if (clang_isExpression(C.kind)) {
    Decl *D = getDeclFromExpr(getCursorExpr(C));
    if (D)
      return getDeclSpelling(D);
    return createCXString("");
  }

  if (clang_isStatement(C.kind)) {
    Stmt *S = getCursorStmt(C);
    if (LabelStmt *Label = dyn_cast_or_null<LabelStmt>(S))
      return createCXString(Label->getName());

    return createCXString("");
  }
  
  if (C.kind == CXCursor_MacroInstantiation)
    return createCXString(getCursorMacroInstantiation(C)->getName()
                                                           ->getNameStart());

  if (C.kind == CXCursor_MacroDefinition)
    return createCXString(getCursorMacroDefinition(C)->getName()
                                                           ->getNameStart());

  if (C.kind == CXCursor_InclusionDirective)
    return createCXString(getCursorInclusionDirective(C)->getFileName());
      
  if (clang_isDeclaration(C.kind))
    return getDeclSpelling(getCursorDecl(C));

  return createCXString("");
}

CXString clang_getCursorDisplayName(CXCursor C) {
  if (!clang_isDeclaration(C.kind))
    return clang_getCursorSpelling(C);
  
  Decl *D = getCursorDecl(C);
  if (!D)
    return createCXString("");

  PrintingPolicy &Policy = getCursorContext(C).PrintingPolicy;
  if (FunctionTemplateDecl *FunTmpl = dyn_cast<FunctionTemplateDecl>(D))
    D = FunTmpl->getTemplatedDecl();
  
  if (FunctionDecl *Function = dyn_cast<FunctionDecl>(D)) {
    llvm::SmallString<64> Str;
    llvm::raw_svector_ostream OS(Str);
    OS << Function->getNameAsString();
    if (Function->getPrimaryTemplate())
      OS << "<>";
    OS << "(";
    for (unsigned I = 0, N = Function->getNumParams(); I != N; ++I) {
      if (I)
        OS << ", ";
      OS << Function->getParamDecl(I)->getType().getAsString(Policy);
    }
    
    if (Function->isVariadic()) {
      if (Function->getNumParams())
        OS << ", ";
      OS << "...";
    }
    OS << ")";
    return createCXString(OS.str());
  }
  
  if (ClassTemplateDecl *ClassTemplate = dyn_cast<ClassTemplateDecl>(D)) {
    llvm::SmallString<64> Str;
    llvm::raw_svector_ostream OS(Str);
    OS << ClassTemplate->getNameAsString();
    OS << "<";
    TemplateParameterList *Params = ClassTemplate->getTemplateParameters();
    for (unsigned I = 0, N = Params->size(); I != N; ++I) {
      if (I)
        OS << ", ";
      
      NamedDecl *Param = Params->getParam(I);
      if (Param->getIdentifier()) {
        OS << Param->getIdentifier()->getName();
        continue;
      }
      
      // There is no parameter name, which makes this tricky. Try to come up
      // with something useful that isn't too long.
      if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(Param))
        OS << (TTP->wasDeclaredWithTypename()? "typename" : "class");
      else if (NonTypeTemplateParmDecl *NTTP
                                    = dyn_cast<NonTypeTemplateParmDecl>(Param))
        OS << NTTP->getType().getAsString(Policy);
      else
        OS << "template<...> class";
    }
    
    OS << ">";
    return createCXString(OS.str());
  }
  
  if (ClassTemplateSpecializationDecl *ClassSpec
                              = dyn_cast<ClassTemplateSpecializationDecl>(D)) {
    // If the type was explicitly written, use that.
    if (TypeSourceInfo *TSInfo = ClassSpec->getTypeAsWritten())
      return createCXString(TSInfo->getType().getAsString(Policy));
    
    llvm::SmallString<64> Str;
    llvm::raw_svector_ostream OS(Str);
    OS << ClassSpec->getNameAsString();
    OS << TemplateSpecializationType::PrintTemplateArgumentList(
                                      ClassSpec->getTemplateArgs().data(),
                                      ClassSpec->getTemplateArgs().size(),
                                                                Policy);
    return createCXString(OS.str());
  }
  
  return clang_getCursorSpelling(C);
}
  
CXString clang_getCursorKindSpelling(enum CXCursorKind Kind) {
  switch (Kind) {
  case CXCursor_FunctionDecl:
      return createCXString("FunctionDecl");
  case CXCursor_TypedefDecl:
      return createCXString("TypedefDecl");
  case CXCursor_EnumDecl:
      return createCXString("EnumDecl");
  case CXCursor_EnumConstantDecl:
      return createCXString("EnumConstantDecl");
  case CXCursor_StructDecl:
      return createCXString("StructDecl");
  case CXCursor_UnionDecl:
      return createCXString("UnionDecl");
  case CXCursor_ClassDecl:
      return createCXString("ClassDecl");
  case CXCursor_FieldDecl:
      return createCXString("FieldDecl");
  case CXCursor_VarDecl:
      return createCXString("VarDecl");
  case CXCursor_ParmDecl:
      return createCXString("ParmDecl");
  case CXCursor_ObjCInterfaceDecl:
      return createCXString("ObjCInterfaceDecl");
  case CXCursor_ObjCCategoryDecl:
      return createCXString("ObjCCategoryDecl");
  case CXCursor_ObjCProtocolDecl:
      return createCXString("ObjCProtocolDecl");
  case CXCursor_ObjCPropertyDecl:
      return createCXString("ObjCPropertyDecl");
  case CXCursor_ObjCIvarDecl:
      return createCXString("ObjCIvarDecl");
  case CXCursor_ObjCInstanceMethodDecl:
      return createCXString("ObjCInstanceMethodDecl");
  case CXCursor_ObjCClassMethodDecl:
      return createCXString("ObjCClassMethodDecl");
  case CXCursor_ObjCImplementationDecl:
      return createCXString("ObjCImplementationDecl");
  case CXCursor_ObjCCategoryImplDecl:
      return createCXString("ObjCCategoryImplDecl");
  case CXCursor_CXXMethod:
      return createCXString("CXXMethod");
  case CXCursor_UnexposedDecl:
      return createCXString("UnexposedDecl");
  case CXCursor_ObjCSuperClassRef:
      return createCXString("ObjCSuperClassRef");
  case CXCursor_ObjCProtocolRef:
      return createCXString("ObjCProtocolRef");
  case CXCursor_ObjCClassRef:
      return createCXString("ObjCClassRef");
  case CXCursor_TypeRef:
      return createCXString("TypeRef");
  case CXCursor_TemplateRef:
      return createCXString("TemplateRef");
  case CXCursor_NamespaceRef:
    return createCXString("NamespaceRef");
  case CXCursor_MemberRef:
    return createCXString("MemberRef");
  case CXCursor_LabelRef:
    return createCXString("LabelRef");
  case CXCursor_OverloadedDeclRef:
    return createCXString("OverloadedDeclRef");
  case CXCursor_UnexposedExpr:
      return createCXString("UnexposedExpr");
  case CXCursor_BlockExpr:
      return createCXString("BlockExpr");
  case CXCursor_DeclRefExpr:
      return createCXString("DeclRefExpr");
  case CXCursor_MemberRefExpr:
      return createCXString("MemberRefExpr");
  case CXCursor_CallExpr:
      return createCXString("CallExpr");
  case CXCursor_ObjCMessageExpr:
      return createCXString("ObjCMessageExpr");
  case CXCursor_UnexposedStmt:
      return createCXString("UnexposedStmt");
  case CXCursor_LabelStmt:
      return createCXString("LabelStmt");
  case CXCursor_InvalidFile:
      return createCXString("InvalidFile");
  case CXCursor_InvalidCode:
    return createCXString("InvalidCode");
  case CXCursor_NoDeclFound:
      return createCXString("NoDeclFound");
  case CXCursor_NotImplemented:
      return createCXString("NotImplemented");
  case CXCursor_TranslationUnit:
      return createCXString("TranslationUnit");
  case CXCursor_UnexposedAttr:
      return createCXString("UnexposedAttr");
  case CXCursor_IBActionAttr:
      return createCXString("attribute(ibaction)");
  case CXCursor_IBOutletAttr:
     return createCXString("attribute(iboutlet)");
  case CXCursor_IBOutletCollectionAttr:
      return createCXString("attribute(iboutletcollection)");
  case CXCursor_PreprocessingDirective:
    return createCXString("preprocessing directive");
  case CXCursor_MacroDefinition:
    return createCXString("macro definition");
  case CXCursor_MacroInstantiation:
    return createCXString("macro instantiation");
  case CXCursor_InclusionDirective:
    return createCXString("inclusion directive");
  case CXCursor_Namespace:
    return createCXString("Namespace");
  case CXCursor_LinkageSpec:
    return createCXString("LinkageSpec");
  case CXCursor_CXXBaseSpecifier:
    return createCXString("C++ base class specifier");  
  case CXCursor_Constructor:
    return createCXString("CXXConstructor");
  case CXCursor_Destructor:
    return createCXString("CXXDestructor");
  case CXCursor_ConversionFunction:
    return createCXString("CXXConversion");
  case CXCursor_TemplateTypeParameter:
    return createCXString("TemplateTypeParameter");
  case CXCursor_NonTypeTemplateParameter:
    return createCXString("NonTypeTemplateParameter");
  case CXCursor_TemplateTemplateParameter:
    return createCXString("TemplateTemplateParameter");
  case CXCursor_FunctionTemplate:
    return createCXString("FunctionTemplate");
  case CXCursor_ClassTemplate:
    return createCXString("ClassTemplate");
  case CXCursor_ClassTemplatePartialSpecialization:
    return createCXString("ClassTemplatePartialSpecialization");
  case CXCursor_NamespaceAlias:
    return createCXString("NamespaceAlias");
  case CXCursor_UsingDirective:
    return createCXString("UsingDirective");
  case CXCursor_UsingDeclaration:
    return createCXString("UsingDeclaration");
  }

  llvm_unreachable("Unhandled CXCursorKind");
  return createCXString((const char*) 0);
}

enum CXChildVisitResult GetCursorVisitor(CXCursor cursor,
                                         CXCursor parent,
                                         CXClientData client_data) {
  CXCursor *BestCursor = static_cast<CXCursor *>(client_data);
  
  // If our current best cursor is the construction of a temporary object, 
  // don't replace that cursor with a type reference, because we want 
  // clang_getCursor() to point at the constructor.
  if (clang_isExpression(BestCursor->kind) &&
      isa<CXXTemporaryObjectExpr>(getCursorExpr(*BestCursor)) &&
      cursor.kind == CXCursor_TypeRef)
    return CXChildVisit_Recurse;
  
  // Don't override a preprocessing cursor with another preprocessing
  // cursor; we want the outermost preprocessing cursor.
  if (clang_isPreprocessing(cursor.kind) &&
      clang_isPreprocessing(BestCursor->kind))
    return CXChildVisit_Recurse;
  
  *BestCursor = cursor;
  return CXChildVisit_Recurse;
}

CXCursor clang_getCursor(CXTranslationUnit TU, CXSourceLocation Loc) {
  if (!TU)
    return clang_getNullCursor();

  ASTUnit *CXXUnit = static_cast<ASTUnit *>(TU->TUData);
  ASTUnit::ConcurrencyCheck Check(*CXXUnit);

  // Translate the given source location to make it point at the beginning of
  // the token under the cursor.
  SourceLocation SLoc = cxloc::translateSourceLocation(Loc);

  // Guard against an invalid SourceLocation, or we may assert in one
  // of the following calls.
  if (SLoc.isInvalid())
    return clang_getNullCursor();

  bool Logging = getenv("LIBCLANG_LOGGING");  
  SLoc = Lexer::GetBeginningOfToken(SLoc, CXXUnit->getSourceManager(),
                                    CXXUnit->getASTContext().getLangOptions());
  
  CXCursor Result = MakeCXCursorInvalid(CXCursor_NoDeclFound);
  if (SLoc.isValid()) {
    // FIXME: Would be great to have a "hint" cursor, then walk from that
    // hint cursor upward until we find a cursor whose source range encloses
    // the region of interest, rather than starting from the translation unit.
    CXCursor Parent = clang_getTranslationUnitCursor(TU);
    CursorVisitor CursorVis(TU, GetCursorVisitor, &Result,
                            Decl::MaxPCHLevel, SourceLocation(SLoc));
    CursorVis.VisitChildren(Parent);
  }
  
  if (Logging) {
    CXFile SearchFile;
    unsigned SearchLine, SearchColumn;
    CXFile ResultFile;
    unsigned ResultLine, ResultColumn;
    CXString SearchFileName, ResultFileName, KindSpelling, USR;
    const char *IsDef = clang_isCursorDefinition(Result)? " (Definition)" : "";
    CXSourceLocation ResultLoc = clang_getCursorLocation(Result);
    
    clang_getInstantiationLocation(Loc, &SearchFile, &SearchLine, &SearchColumn,
                                   0);
    clang_getInstantiationLocation(ResultLoc, &ResultFile, &ResultLine, 
                                   &ResultColumn, 0);
    SearchFileName = clang_getFileName(SearchFile);
    ResultFileName = clang_getFileName(ResultFile);
    KindSpelling = clang_getCursorKindSpelling(Result.kind);
    USR = clang_getCursorUSR(Result);
    fprintf(stderr, "clang_getCursor(%s:%d:%d) = %s(%s:%d:%d):%s%s\n",
            clang_getCString(SearchFileName), SearchLine, SearchColumn,
            clang_getCString(KindSpelling),
            clang_getCString(ResultFileName), ResultLine, ResultColumn,
            clang_getCString(USR), IsDef);
    clang_disposeString(SearchFileName);
    clang_disposeString(ResultFileName);
    clang_disposeString(KindSpelling);
    clang_disposeString(USR);
    
    CXCursor Definition = clang_getCursorDefinition(Result);
    if (!clang_equalCursors(Definition, clang_getNullCursor())) {
      CXSourceLocation DefinitionLoc = clang_getCursorLocation(Definition);
      CXString DefinitionKindSpelling
                                = clang_getCursorKindSpelling(Definition.kind);
      CXFile DefinitionFile;
      unsigned DefinitionLine, DefinitionColumn;
      clang_getInstantiationLocation(DefinitionLoc, &DefinitionFile, 
                                     &DefinitionLine, &DefinitionColumn, 0);
      CXString DefinitionFileName = clang_getFileName(DefinitionFile);
      fprintf(stderr, "  -> %s(%s:%d:%d)\n",
              clang_getCString(DefinitionKindSpelling),
              clang_getCString(DefinitionFileName),
              DefinitionLine, DefinitionColumn);
      clang_disposeString(DefinitionFileName);
      clang_disposeString(DefinitionKindSpelling);
    }
  }

  return Result;
}

CXCursor clang_getNullCursor(void) {
  return MakeCXCursorInvalid(CXCursor_InvalidFile);
}

unsigned clang_equalCursors(CXCursor X, CXCursor Y) {
  return X == Y;
}

unsigned clang_hashCursor(CXCursor C) {
  unsigned Index = 0;
  if (clang_isExpression(C.kind) || clang_isStatement(C.kind))
    Index = 1;
  
  return llvm::DenseMapInfo<std::pair<unsigned, void*> >::getHashValue(
                                        std::make_pair(C.kind, C.data[Index]));
}

unsigned clang_isInvalid(enum CXCursorKind K) {
  return K >= CXCursor_FirstInvalid && K <= CXCursor_LastInvalid;
}

unsigned clang_isDeclaration(enum CXCursorKind K) {
  return K >= CXCursor_FirstDecl && K <= CXCursor_LastDecl;
}

unsigned clang_isReference(enum CXCursorKind K) {
  return K >= CXCursor_FirstRef && K <= CXCursor_LastRef;
}

unsigned clang_isExpression(enum CXCursorKind K) {
  return K >= CXCursor_FirstExpr && K <= CXCursor_LastExpr;
}

unsigned clang_isStatement(enum CXCursorKind K) {
  return K >= CXCursor_FirstStmt && K <= CXCursor_LastStmt;
}

unsigned clang_isTranslationUnit(enum CXCursorKind K) {
  return K == CXCursor_TranslationUnit;
}

unsigned clang_isPreprocessing(enum CXCursorKind K) {
  return K >= CXCursor_FirstPreprocessing && K <= CXCursor_LastPreprocessing;
}
  
unsigned clang_isUnexposed(enum CXCursorKind K) {
  switch (K) {
    case CXCursor_UnexposedDecl:
    case CXCursor_UnexposedExpr:
    case CXCursor_UnexposedStmt:
    case CXCursor_UnexposedAttr:
      return true;
    default:
      return false;
  }
}

CXCursorKind clang_getCursorKind(CXCursor C) {
  return C.kind;
}

CXSourceLocation clang_getCursorLocation(CXCursor C) {
  if (clang_isReference(C.kind)) {
    switch (C.kind) {
    case CXCursor_ObjCSuperClassRef: {
      std::pair<ObjCInterfaceDecl *, SourceLocation> P
        = getCursorObjCSuperClassRef(C);
      return cxloc::translateSourceLocation(P.first->getASTContext(), P.second);
    }

    case CXCursor_ObjCProtocolRef: {
      std::pair<ObjCProtocolDecl *, SourceLocation> P
        = getCursorObjCProtocolRef(C);
      return cxloc::translateSourceLocation(P.first->getASTContext(), P.second);
    }

    case CXCursor_ObjCClassRef: {
      std::pair<ObjCInterfaceDecl *, SourceLocation> P
        = getCursorObjCClassRef(C);
      return cxloc::translateSourceLocation(P.first->getASTContext(), P.second);
    }

    case CXCursor_TypeRef: {
      std::pair<TypeDecl *, SourceLocation> P = getCursorTypeRef(C);
      return cxloc::translateSourceLocation(P.first->getASTContext(), P.second);
    }

    case CXCursor_TemplateRef: {
      std::pair<TemplateDecl *, SourceLocation> P = getCursorTemplateRef(C);
      return cxloc::translateSourceLocation(P.first->getASTContext(), P.second);
    }

    case CXCursor_NamespaceRef: {
      std::pair<NamedDecl *, SourceLocation> P = getCursorNamespaceRef(C);
      return cxloc::translateSourceLocation(P.first->getASTContext(), P.second);
    }

    case CXCursor_MemberRef: {
      std::pair<FieldDecl *, SourceLocation> P = getCursorMemberRef(C);
      return cxloc::translateSourceLocation(P.first->getASTContext(), P.second);
    }

    case CXCursor_CXXBaseSpecifier: {
      CXXBaseSpecifier *BaseSpec = getCursorCXXBaseSpecifier(C);
      if (!BaseSpec)
        return clang_getNullLocation();
      
      if (TypeSourceInfo *TSInfo = BaseSpec->getTypeSourceInfo())
        return cxloc::translateSourceLocation(getCursorContext(C),
                                            TSInfo->getTypeLoc().getBeginLoc());
      
      return cxloc::translateSourceLocation(getCursorContext(C),
                                        BaseSpec->getSourceRange().getBegin());
    }

    case CXCursor_LabelRef: {
      std::pair<LabelStmt *, SourceLocation> P = getCursorLabelRef(C);
      return cxloc::translateSourceLocation(getCursorContext(C), P.second);
    }

    case CXCursor_OverloadedDeclRef:
      return cxloc::translateSourceLocation(getCursorContext(C),
                                          getCursorOverloadedDeclRef(C).second);

    default:
      // FIXME: Need a way to enumerate all non-reference cases.
      llvm_unreachable("Missed a reference kind");
    }
  }

  if (clang_isExpression(C.kind))
    return cxloc::translateSourceLocation(getCursorContext(C),
                                   getLocationFromExpr(getCursorExpr(C)));

  if (clang_isStatement(C.kind))
    return cxloc::translateSourceLocation(getCursorContext(C),
                                          getCursorStmt(C)->getLocStart());

  if (C.kind == CXCursor_PreprocessingDirective) {
    SourceLocation L = cxcursor::getCursorPreprocessingDirective(C).getBegin();
    return cxloc::translateSourceLocation(getCursorContext(C), L);
  }

  if (C.kind == CXCursor_MacroInstantiation) {
    SourceLocation L
      = cxcursor::getCursorMacroInstantiation(C)->getSourceRange().getBegin();
    return cxloc::translateSourceLocation(getCursorContext(C), L);
  }

  if (C.kind == CXCursor_MacroDefinition) {
    SourceLocation L = cxcursor::getCursorMacroDefinition(C)->getLocation();
    return cxloc::translateSourceLocation(getCursorContext(C), L);
  }

  if (C.kind == CXCursor_InclusionDirective) {
    SourceLocation L
      = cxcursor::getCursorInclusionDirective(C)->getSourceRange().getBegin();
    return cxloc::translateSourceLocation(getCursorContext(C), L);
  }

  if (C.kind < CXCursor_FirstDecl || C.kind > CXCursor_LastDecl)
    return clang_getNullLocation();

  Decl *D = getCursorDecl(C);
  SourceLocation Loc = D->getLocation();
  if (ObjCInterfaceDecl *Class = dyn_cast<ObjCInterfaceDecl>(D))
    Loc = Class->getClassLoc();
  // FIXME: Multiple variables declared in a single declaration
  // currently lack the information needed to correctly determine their
  // ranges when accounting for the type-specifier.  We use context
  // stored in the CXCursor to determine if the VarDecl is in a DeclGroup,
  // and if so, whether it is the first decl.
  if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
    if (!cxcursor::isFirstInDeclGroup(C))
      Loc = VD->getLocation();
  }

  return cxloc::translateSourceLocation(getCursorContext(C), Loc);
}

} // end extern "C"

static SourceRange getRawCursorExtent(CXCursor C) {
  if (clang_isReference(C.kind)) {
    switch (C.kind) {
    case CXCursor_ObjCSuperClassRef:
      return  getCursorObjCSuperClassRef(C).second;

    case CXCursor_ObjCProtocolRef:
      return getCursorObjCProtocolRef(C).second;

    case CXCursor_ObjCClassRef:
      return getCursorObjCClassRef(C).second;

    case CXCursor_TypeRef:
      return getCursorTypeRef(C).second;

    case CXCursor_TemplateRef:
      return getCursorTemplateRef(C).second;

    case CXCursor_NamespaceRef:
      return getCursorNamespaceRef(C).second;

    case CXCursor_MemberRef:
      return getCursorMemberRef(C).second;

    case CXCursor_CXXBaseSpecifier:
      return getCursorCXXBaseSpecifier(C)->getSourceRange();

    case CXCursor_LabelRef:
      return getCursorLabelRef(C).second;

    case CXCursor_OverloadedDeclRef:
      return getCursorOverloadedDeclRef(C).second;

    default:
      // FIXME: Need a way to enumerate all non-reference cases.
      llvm_unreachable("Missed a reference kind");
    }
  }

  if (clang_isExpression(C.kind))
    return getCursorExpr(C)->getSourceRange();

  if (clang_isStatement(C.kind))
    return getCursorStmt(C)->getSourceRange();

  if (C.kind == CXCursor_PreprocessingDirective)
    return cxcursor::getCursorPreprocessingDirective(C);

  if (C.kind == CXCursor_MacroInstantiation)
    return cxcursor::getCursorMacroInstantiation(C)->getSourceRange();

  if (C.kind == CXCursor_MacroDefinition)
    return cxcursor::getCursorMacroDefinition(C)->getSourceRange();

  if (C.kind == CXCursor_InclusionDirective)
    return cxcursor::getCursorInclusionDirective(C)->getSourceRange();

  if (C.kind >= CXCursor_FirstDecl && C.kind <= CXCursor_LastDecl) {
    Decl *D = cxcursor::getCursorDecl(C);
    SourceRange R = D->getSourceRange();
    // FIXME: Multiple variables declared in a single declaration
    // currently lack the information needed to correctly determine their
    // ranges when accounting for the type-specifier.  We use context
    // stored in the CXCursor to determine if the VarDecl is in a DeclGroup,
    // and if so, whether it is the first decl.
    if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
      if (!cxcursor::isFirstInDeclGroup(C))
        R.setBegin(VD->getLocation());
    }
    return R;
  }
  return SourceRange();
}

/// \brief Retrieves the "raw" cursor extent, which is then extended to include
/// the decl-specifier-seq for declarations.
static SourceRange getFullCursorExtent(CXCursor C, SourceManager &SrcMgr) {
  if (C.kind >= CXCursor_FirstDecl && C.kind <= CXCursor_LastDecl) {
    Decl *D = cxcursor::getCursorDecl(C);
    SourceRange R = D->getSourceRange();

    // Adjust the start of the location for declarations preceded by
    // declaration specifiers.
    SourceLocation StartLoc;
    if (const DeclaratorDecl *DD = dyn_cast<DeclaratorDecl>(D)) {
      if (TypeSourceInfo *TI = DD->getTypeSourceInfo())
        StartLoc = TI->getTypeLoc().getSourceRange().getBegin();
    } else if (TypedefDecl *Typedef = dyn_cast<TypedefDecl>(D)) {
      if (TypeSourceInfo *TI = Typedef->getTypeSourceInfo())
        StartLoc = TI->getTypeLoc().getSourceRange().getBegin();
    }

    if (StartLoc.isValid() && R.getBegin().isValid() &&
        SrcMgr.isBeforeInTranslationUnit(StartLoc, R.getBegin()))
      R.setBegin(StartLoc);

    // FIXME: Multiple variables declared in a single declaration
    // currently lack the information needed to correctly determine their
    // ranges when accounting for the type-specifier.  We use context
    // stored in the CXCursor to determine if the VarDecl is in a DeclGroup,
    // and if so, whether it is the first decl.
    if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
      if (!cxcursor::isFirstInDeclGroup(C))
        R.setBegin(VD->getLocation());
    }

    return R;    
  }
  
  return getRawCursorExtent(C);
}

extern "C" {

CXSourceRange clang_getCursorExtent(CXCursor C) {
  SourceRange R = getRawCursorExtent(C);
  if (R.isInvalid())
    return clang_getNullRange();

  return cxloc::translateSourceRange(getCursorContext(C), R);
}

CXCursor clang_getCursorReferenced(CXCursor C) {
  if (clang_isInvalid(C.kind))
    return clang_getNullCursor();

  CXTranslationUnit tu = getCursorTU(C);
  if (clang_isDeclaration(C.kind)) {
    Decl *D = getCursorDecl(C);
    if (UsingDecl *Using = dyn_cast<UsingDecl>(D))
      return MakeCursorOverloadedDeclRef(Using, D->getLocation(), tu);
    if (ObjCClassDecl *Classes = dyn_cast<ObjCClassDecl>(D))
      return MakeCursorOverloadedDeclRef(Classes, D->getLocation(), tu);
    if (ObjCForwardProtocolDecl *Protocols
                                        = dyn_cast<ObjCForwardProtocolDecl>(D))
      return MakeCursorOverloadedDeclRef(Protocols, D->getLocation(), tu);
    if (ObjCPropertyImplDecl *PropImpl =llvm::dyn_cast<ObjCPropertyImplDecl>(D))
      if (ObjCPropertyDecl *Property = PropImpl->getPropertyDecl())
        return MakeCXCursor(Property, tu);
    
    return C;
  }
  
  if (clang_isExpression(C.kind)) {
    Expr *E = getCursorExpr(C);
    Decl *D = getDeclFromExpr(E);
    if (D)
      return MakeCXCursor(D, tu);
    
    if (OverloadExpr *Ovl = dyn_cast_or_null<OverloadExpr>(E))
      return MakeCursorOverloadedDeclRef(Ovl, tu);
        
    return clang_getNullCursor();
  }

  if (clang_isStatement(C.kind)) {
    Stmt *S = getCursorStmt(C);
    if (GotoStmt *Goto = dyn_cast_or_null<GotoStmt>(S))
      return MakeCXCursor(Goto->getLabel()->getStmt(), getCursorDecl(C), tu);

    return clang_getNullCursor();
  }
  
  if (C.kind == CXCursor_MacroInstantiation) {
    if (MacroDefinition *Def = getCursorMacroInstantiation(C)->getDefinition())
      return MakeMacroDefinitionCursor(Def, tu);
  }

  if (!clang_isReference(C.kind))
    return clang_getNullCursor();

  switch (C.kind) {
    case CXCursor_ObjCSuperClassRef:
      return MakeCXCursor(getCursorObjCSuperClassRef(C).first, tu);

    case CXCursor_ObjCProtocolRef: {
      return MakeCXCursor(getCursorObjCProtocolRef(C).first, tu);

    case CXCursor_ObjCClassRef:
      return MakeCXCursor(getCursorObjCClassRef(C).first, tu );

    case CXCursor_TypeRef:
      return MakeCXCursor(getCursorTypeRef(C).first, tu );

    case CXCursor_TemplateRef:
      return MakeCXCursor(getCursorTemplateRef(C).first, tu );

    case CXCursor_NamespaceRef:
      return MakeCXCursor(getCursorNamespaceRef(C).first, tu );

    case CXCursor_MemberRef:
      return MakeCXCursor(getCursorMemberRef(C).first, tu );

    case CXCursor_CXXBaseSpecifier: {
      CXXBaseSpecifier *B = cxcursor::getCursorCXXBaseSpecifier(C);
      return clang_getTypeDeclaration(cxtype::MakeCXType(B->getType(),
                                                         tu ));
    }

    case CXCursor_LabelRef:
      // FIXME: We end up faking the "parent" declaration here because we
      // don't want to make CXCursor larger.
      return MakeCXCursor(getCursorLabelRef(C).first, 
               static_cast<ASTUnit*>(tu->TUData)->getASTContext()
                          .getTranslationUnitDecl(),
                          tu);

    case CXCursor_OverloadedDeclRef:
      return C;

    default:
      // We would prefer to enumerate all non-reference cursor kinds here.
      llvm_unreachable("Unhandled reference cursor kind");
      break;
    }
  }

  return clang_getNullCursor();
}

CXCursor clang_getCursorDefinition(CXCursor C) {
  if (clang_isInvalid(C.kind))
    return clang_getNullCursor();

  CXTranslationUnit TU = getCursorTU(C);

  bool WasReference = false;
  if (clang_isReference(C.kind) || clang_isExpression(C.kind)) {
    C = clang_getCursorReferenced(C);
    WasReference = true;
  }

  if (C.kind == CXCursor_MacroInstantiation)
    return clang_getCursorReferenced(C);

  if (!clang_isDeclaration(C.kind))
    return clang_getNullCursor();

  Decl *D = getCursorDecl(C);
  if (!D)
    return clang_getNullCursor();

  switch (D->getKind()) {
  // Declaration kinds that don't really separate the notions of
  // declaration and definition.
  case Decl::Namespace:
  case Decl::Typedef:
  case Decl::TemplateTypeParm:
  case Decl::EnumConstant:
  case Decl::Field:
  case Decl::IndirectField:
  case Decl::ObjCIvar:
  case Decl::ObjCAtDefsField:
  case Decl::ImplicitParam:
  case Decl::ParmVar:
  case Decl::NonTypeTemplateParm:
  case Decl::TemplateTemplateParm:
  case Decl::ObjCCategoryImpl:
  case Decl::ObjCImplementation:
  case Decl::AccessSpec:
  case Decl::LinkageSpec:
  case Decl::ObjCPropertyImpl:
  case Decl::FileScopeAsm:
  case Decl::StaticAssert:
  case Decl::Block:
  case Decl::Label:  // FIXME: Is this right??
    return C;

  // Declaration kinds that don't make any sense here, but are
  // nonetheless harmless.
  case Decl::TranslationUnit:
    break;

  // Declaration kinds for which the definition is not resolvable.
  case Decl::UnresolvedUsingTypename:
  case Decl::UnresolvedUsingValue:
    break;

  case Decl::UsingDirective:
    return MakeCXCursor(cast<UsingDirectiveDecl>(D)->getNominatedNamespace(),
                        TU);

  case Decl::NamespaceAlias:
    return MakeCXCursor(cast<NamespaceAliasDecl>(D)->getNamespace(), TU);

  case Decl::Enum:
  case Decl::Record:
  case Decl::CXXRecord:
  case Decl::ClassTemplateSpecialization:
  case Decl::ClassTemplatePartialSpecialization:
    if (TagDecl *Def = cast<TagDecl>(D)->getDefinition())
      return MakeCXCursor(Def, TU);
    return clang_getNullCursor();

  case Decl::Function:
  case Decl::CXXMethod:
  case Decl::CXXConstructor:
  case Decl::CXXDestructor:
  case Decl::CXXConversion: {
    const FunctionDecl *Def = 0;
    if (cast<FunctionDecl>(D)->getBody(Def))
      return MakeCXCursor(const_cast<FunctionDecl *>(Def), TU);
    return clang_getNullCursor();
  }

  case Decl::Var: {
    // Ask the variable if it has a definition.
    if (VarDecl *Def = cast<VarDecl>(D)->getDefinition())
      return MakeCXCursor(Def, TU);
    return clang_getNullCursor();
  }

  case Decl::FunctionTemplate: {
    const FunctionDecl *Def = 0;
    if (cast<FunctionTemplateDecl>(D)->getTemplatedDecl()->getBody(Def))
      return MakeCXCursor(Def->getDescribedFunctionTemplate(), TU);
    return clang_getNullCursor();
  }

  case Decl::ClassTemplate: {
    if (RecordDecl *Def = cast<ClassTemplateDecl>(D)->getTemplatedDecl()
                                                            ->getDefinition())
      return MakeCXCursor(cast<CXXRecordDecl>(Def)->getDescribedClassTemplate(),
                          TU);
    return clang_getNullCursor();
  }

  case Decl::Using:
    return MakeCursorOverloadedDeclRef(cast<UsingDecl>(D), 
                                       D->getLocation(), TU);

  case Decl::UsingShadow:
    return clang_getCursorDefinition(
                       MakeCXCursor(cast<UsingShadowDecl>(D)->getTargetDecl(),
                                    TU));

  case Decl::ObjCMethod: {
    ObjCMethodDecl *Method = cast<ObjCMethodDecl>(D);
    if (Method->isThisDeclarationADefinition())
      return C;

    // Dig out the method definition in the associated
    // @implementation, if we have it.
    // FIXME: The ASTs should make finding the definition easier.
    if (ObjCInterfaceDecl *Class
                       = dyn_cast<ObjCInterfaceDecl>(Method->getDeclContext()))
      if (ObjCImplementationDecl *ClassImpl = Class->getImplementation())
        if (ObjCMethodDecl *Def = ClassImpl->getMethod(Method->getSelector(),
                                                  Method->isInstanceMethod()))
          if (Def->isThisDeclarationADefinition())
            return MakeCXCursor(Def, TU);

    return clang_getNullCursor();
  }

  case Decl::ObjCCategory:
    if (ObjCCategoryImplDecl *Impl
                               = cast<ObjCCategoryDecl>(D)->getImplementation())
      return MakeCXCursor(Impl, TU);
    return clang_getNullCursor();

  case Decl::ObjCProtocol:
    if (!cast<ObjCProtocolDecl>(D)->isForwardDecl())
      return C;
    return clang_getNullCursor();

  case Decl::ObjCInterface:
    // There are two notions of a "definition" for an Objective-C
    // class: the interface and its implementation. When we resolved a
    // reference to an Objective-C class, produce the @interface as
    // the definition; when we were provided with the interface,
    // produce the @implementation as the definition.
    if (WasReference) {
      if (!cast<ObjCInterfaceDecl>(D)->isForwardDecl())
        return C;
    } else if (ObjCImplementationDecl *Impl
                              = cast<ObjCInterfaceDecl>(D)->getImplementation())
      return MakeCXCursor(Impl, TU);
    return clang_getNullCursor();

  case Decl::ObjCProperty:
    // FIXME: We don't really know where to find the
    // ObjCPropertyImplDecls that implement this property.
    return clang_getNullCursor();

  case Decl::ObjCCompatibleAlias:
    if (ObjCInterfaceDecl *Class
          = cast<ObjCCompatibleAliasDecl>(D)->getClassInterface())
      if (!Class->isForwardDecl())
        return MakeCXCursor(Class, TU);

    return clang_getNullCursor();

  case Decl::ObjCForwardProtocol:
    return MakeCursorOverloadedDeclRef(cast<ObjCForwardProtocolDecl>(D), 
                                       D->getLocation(), TU);

  case Decl::ObjCClass:
    return MakeCursorOverloadedDeclRef(cast<ObjCClassDecl>(D), D->getLocation(),
                                       TU);

  case Decl::Friend:
    if (NamedDecl *Friend = cast<FriendDecl>(D)->getFriendDecl())
      return clang_getCursorDefinition(MakeCXCursor(Friend, TU));
    return clang_getNullCursor();

  case Decl::FriendTemplate:
    if (NamedDecl *Friend = cast<FriendTemplateDecl>(D)->getFriendDecl())
      return clang_getCursorDefinition(MakeCXCursor(Friend, TU));
    return clang_getNullCursor();
  }

  return clang_getNullCursor();
}

unsigned clang_isCursorDefinition(CXCursor C) {
  if (!clang_isDeclaration(C.kind))
    return 0;

  return clang_getCursorDefinition(C) == C;
}

CXCursor clang_getCanonicalCursor(CXCursor C) {
  if (!clang_isDeclaration(C.kind))
    return C;
  
  if (Decl *D = getCursorDecl(C))
    return MakeCXCursor(D->getCanonicalDecl(), getCursorTU(C));
  
  return C;
}
  
unsigned clang_getNumOverloadedDecls(CXCursor C) {
  if (C.kind != CXCursor_OverloadedDeclRef)
    return 0;
  
  OverloadedDeclRefStorage Storage = getCursorOverloadedDeclRef(C).first;
  if (OverloadExpr *E = Storage.dyn_cast<OverloadExpr *>())
    return E->getNumDecls();
  
  if (OverloadedTemplateStorage *S
                              = Storage.dyn_cast<OverloadedTemplateStorage*>())
    return S->size();
  
  Decl *D = Storage.get<Decl*>();
  if (UsingDecl *Using = dyn_cast<UsingDecl>(D))
    return Using->shadow_size();
  if (ObjCClassDecl *Classes = dyn_cast<ObjCClassDecl>(D))
    return Classes->size();
  if (ObjCForwardProtocolDecl *Protocols =dyn_cast<ObjCForwardProtocolDecl>(D))
    return Protocols->protocol_size();
  
  return 0;
}

CXCursor clang_getOverloadedDecl(CXCursor cursor, unsigned index) {
  if (cursor.kind != CXCursor_OverloadedDeclRef)
    return clang_getNullCursor();

  if (index >= clang_getNumOverloadedDecls(cursor))
    return clang_getNullCursor();
  
  CXTranslationUnit TU = getCursorTU(cursor);
  OverloadedDeclRefStorage Storage = getCursorOverloadedDeclRef(cursor).first;
  if (OverloadExpr *E = Storage.dyn_cast<OverloadExpr *>())
    return MakeCXCursor(E->decls_begin()[index], TU);
  
  if (OverloadedTemplateStorage *S
                              = Storage.dyn_cast<OverloadedTemplateStorage*>())
    return MakeCXCursor(S->begin()[index], TU);
  
  Decl *D = Storage.get<Decl*>();
  if (UsingDecl *Using = dyn_cast<UsingDecl>(D)) {
    // FIXME: This is, unfortunately, linear time.
    UsingDecl::shadow_iterator Pos = Using->shadow_begin();
    std::advance(Pos, index);
    return MakeCXCursor(cast<UsingShadowDecl>(*Pos)->getTargetDecl(), TU);
  }
  
  if (ObjCClassDecl *Classes = dyn_cast<ObjCClassDecl>(D))
    return MakeCXCursor(Classes->begin()[index].getInterface(), TU);
  
  if (ObjCForwardProtocolDecl *Protocols = dyn_cast<ObjCForwardProtocolDecl>(D))
    return MakeCXCursor(Protocols->protocol_begin()[index], TU);
  
  return clang_getNullCursor();
}
  
void clang_getDefinitionSpellingAndExtent(CXCursor C,
                                          const char **startBuf,
                                          const char **endBuf,
                                          unsigned *startLine,
                                          unsigned *startColumn,
                                          unsigned *endLine,
                                          unsigned *endColumn) {
  assert(getCursorDecl(C) && "CXCursor has null decl");
  NamedDecl *ND = static_cast<NamedDecl *>(getCursorDecl(C));
  FunctionDecl *FD = dyn_cast<FunctionDecl>(ND);
  CompoundStmt *Body = dyn_cast<CompoundStmt>(FD->getBody());

  SourceManager &SM = FD->getASTContext().getSourceManager();
  *startBuf = SM.getCharacterData(Body->getLBracLoc());
  *endBuf = SM.getCharacterData(Body->getRBracLoc());
  *startLine = SM.getSpellingLineNumber(Body->getLBracLoc());
  *startColumn = SM.getSpellingColumnNumber(Body->getLBracLoc());
  *endLine = SM.getSpellingLineNumber(Body->getRBracLoc());
  *endColumn = SM.getSpellingColumnNumber(Body->getRBracLoc());
}

void clang_enableStackTraces(void) {
  llvm::sys::PrintStackTraceOnErrorSignal();
}

void clang_executeOnThread(void (*fn)(void*), void *user_data,
                           unsigned stack_size) {
  llvm::llvm_execute_on_thread(fn, user_data, stack_size);
}

} // end: extern "C"

//===----------------------------------------------------------------------===//
// Token-based Operations.
//===----------------------------------------------------------------------===//

/* CXToken layout:
 *   int_data[0]: a CXTokenKind
 *   int_data[1]: starting token location
 *   int_data[2]: token length
 *   int_data[3]: reserved
 *   ptr_data: for identifiers and keywords, an IdentifierInfo*.
 *   otherwise unused.
 */
extern "C" {

CXTokenKind clang_getTokenKind(CXToken CXTok) {
  return static_cast<CXTokenKind>(CXTok.int_data[0]);
}

CXString clang_getTokenSpelling(CXTranslationUnit TU, CXToken CXTok) {
  switch (clang_getTokenKind(CXTok)) {
  case CXToken_Identifier:
  case CXToken_Keyword:
    // We know we have an IdentifierInfo*, so use that.
    return createCXString(static_cast<IdentifierInfo *>(CXTok.ptr_data)
                            ->getNameStart());

  case CXToken_Literal: {
    // We have stashed the starting pointer in the ptr_data field. Use it.
    const char *Text = static_cast<const char *>(CXTok.ptr_data);
    return createCXString(llvm::StringRef(Text, CXTok.int_data[2]));
  }

  case CXToken_Punctuation:
  case CXToken_Comment:
    break;
  }

  // We have to find the starting buffer pointer the hard way, by
  // deconstructing the source location.
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(TU->TUData);
  if (!CXXUnit)
    return createCXString("");

  SourceLocation Loc = SourceLocation::getFromRawEncoding(CXTok.int_data[1]);
  std::pair<FileID, unsigned> LocInfo
    = CXXUnit->getSourceManager().getDecomposedLoc(Loc);
  bool Invalid = false;
  llvm::StringRef Buffer
    = CXXUnit->getSourceManager().getBufferData(LocInfo.first, &Invalid);
  if (Invalid)
    return createCXString("");

  return createCXString(Buffer.substr(LocInfo.second, CXTok.int_data[2]));
}

CXSourceLocation clang_getTokenLocation(CXTranslationUnit TU, CXToken CXTok) {
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(TU->TUData);
  if (!CXXUnit)
    return clang_getNullLocation();

  return cxloc::translateSourceLocation(CXXUnit->getASTContext(),
                        SourceLocation::getFromRawEncoding(CXTok.int_data[1]));
}

CXSourceRange clang_getTokenExtent(CXTranslationUnit TU, CXToken CXTok) {
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(TU->TUData);
  if (!CXXUnit)
    return clang_getNullRange();

  return cxloc::translateSourceRange(CXXUnit->getASTContext(),
                        SourceLocation::getFromRawEncoding(CXTok.int_data[1]));
}

void clang_tokenize(CXTranslationUnit TU, CXSourceRange Range,
                    CXToken **Tokens, unsigned *NumTokens) {
  if (Tokens)
    *Tokens = 0;
  if (NumTokens)
    *NumTokens = 0;

  ASTUnit *CXXUnit = static_cast<ASTUnit *>(TU->TUData);
  if (!CXXUnit || !Tokens || !NumTokens)
    return;

  ASTUnit::ConcurrencyCheck Check(*CXXUnit);
  
  SourceRange R = cxloc::translateCXSourceRange(Range);
  if (R.isInvalid())
    return;

  SourceManager &SourceMgr = CXXUnit->getSourceManager();
  std::pair<FileID, unsigned> BeginLocInfo
    = SourceMgr.getDecomposedLoc(R.getBegin());
  std::pair<FileID, unsigned> EndLocInfo
    = SourceMgr.getDecomposedLoc(R.getEnd());

  // Cannot tokenize across files.
  if (BeginLocInfo.first != EndLocInfo.first)
    return;

  // Create a lexer
  bool Invalid = false;
  llvm::StringRef Buffer
    = SourceMgr.getBufferData(BeginLocInfo.first, &Invalid);
  if (Invalid)
    return;
  
  Lexer Lex(SourceMgr.getLocForStartOfFile(BeginLocInfo.first),
            CXXUnit->getASTContext().getLangOptions(),
            Buffer.begin(), Buffer.data() + BeginLocInfo.second, Buffer.end());
  Lex.SetCommentRetentionState(true);

  // Lex tokens until we hit the end of the range.
  const char *EffectiveBufferEnd = Buffer.data() + EndLocInfo.second;
  llvm::SmallVector<CXToken, 32> CXTokens;
  Token Tok;
  bool previousWasAt = false;
  do {
    // Lex the next token
    Lex.LexFromRawLexer(Tok);
    if (Tok.is(tok::eof))
      break;

    // Initialize the CXToken.
    CXToken CXTok;

    //   - Common fields
    CXTok.int_data[1] = Tok.getLocation().getRawEncoding();
    CXTok.int_data[2] = Tok.getLength();
    CXTok.int_data[3] = 0;

    //   - Kind-specific fields
    if (Tok.isLiteral()) {
      CXTok.int_data[0] = CXToken_Literal;
      CXTok.ptr_data = (void *)Tok.getLiteralData();
    } else if (Tok.is(tok::raw_identifier)) {
      // Lookup the identifier to determine whether we have a keyword.
      IdentifierInfo *II
        = CXXUnit->getPreprocessor().LookUpIdentifierInfo(Tok);

      if ((II->getObjCKeywordID() != tok::objc_not_keyword) && previousWasAt) {
        CXTok.int_data[0] = CXToken_Keyword;
      }
      else {
        CXTok.int_data[0] = Tok.is(tok::identifier)
          ? CXToken_Identifier
          : CXToken_Keyword;
      }
      CXTok.ptr_data = II;
    } else if (Tok.is(tok::comment)) {
      CXTok.int_data[0] = CXToken_Comment;
      CXTok.ptr_data = 0;
    } else {
      CXTok.int_data[0] = CXToken_Punctuation;
      CXTok.ptr_data = 0;
    }
    CXTokens.push_back(CXTok);
    previousWasAt = Tok.is(tok::at);
  } while (Lex.getBufferLocation() <= EffectiveBufferEnd);

  if (CXTokens.empty())
    return;

  *Tokens = (CXToken *)malloc(sizeof(CXToken) * CXTokens.size());
  memmove(*Tokens, CXTokens.data(), sizeof(CXToken) * CXTokens.size());
  *NumTokens = CXTokens.size();
}

void clang_disposeTokens(CXTranslationUnit TU,
                         CXToken *Tokens, unsigned NumTokens) {
  free(Tokens);
}

} // end: extern "C"

//===----------------------------------------------------------------------===//
// Token annotation APIs.
//===----------------------------------------------------------------------===//

typedef llvm::DenseMap<unsigned, CXCursor> AnnotateTokensData;
static enum CXChildVisitResult AnnotateTokensVisitor(CXCursor cursor,
                                                     CXCursor parent,
                                                     CXClientData client_data);
namespace {
class AnnotateTokensWorker {
  AnnotateTokensData &Annotated;
  CXToken *Tokens;
  CXCursor *Cursors;
  unsigned NumTokens;
  unsigned TokIdx;
  unsigned PreprocessingTokIdx;
  CursorVisitor AnnotateVis;
  SourceManager &SrcMgr;

  bool MoreTokens() const { return TokIdx < NumTokens; }
  unsigned NextToken() const { return TokIdx; }
  void AdvanceToken() { ++TokIdx; }
  SourceLocation GetTokenLoc(unsigned tokI) {
    return SourceLocation::getFromRawEncoding(Tokens[tokI].int_data[1]);
  }

public:
  AnnotateTokensWorker(AnnotateTokensData &annotated,
                       CXToken *tokens, CXCursor *cursors, unsigned numTokens,
                       CXTranslationUnit tu, SourceRange RegionOfInterest)
    : Annotated(annotated), Tokens(tokens), Cursors(cursors),
      NumTokens(numTokens), TokIdx(0), PreprocessingTokIdx(0),
      AnnotateVis(tu,
                  AnnotateTokensVisitor, this,
                  Decl::MaxPCHLevel, RegionOfInterest),
      SrcMgr(static_cast<ASTUnit*>(tu->TUData)->getSourceManager()) {}

  void VisitChildren(CXCursor C) { AnnotateVis.VisitChildren(C); }
  enum CXChildVisitResult Visit(CXCursor cursor, CXCursor parent);
  void AnnotateTokens(CXCursor parent);
  void AnnotateTokens() {
    AnnotateTokens(clang_getTranslationUnitCursor(AnnotateVis.getTU()));
  }
};
}

void AnnotateTokensWorker::AnnotateTokens(CXCursor parent) {
  // Walk the AST within the region of interest, annotating tokens
  // along the way.
  VisitChildren(parent);

  for (unsigned I = 0 ; I < TokIdx ; ++I) {
    AnnotateTokensData::iterator Pos = Annotated.find(Tokens[I].int_data[1]);
    if (Pos != Annotated.end() && 
        (clang_isInvalid(Cursors[I].kind) ||
         Pos->second.kind != CXCursor_PreprocessingDirective))
      Cursors[I] = Pos->second;
  }

  // Finish up annotating any tokens left.
  if (!MoreTokens())
    return;

  const CXCursor &C = clang_getNullCursor();
  for (unsigned I = TokIdx ; I < NumTokens ; ++I) {
    AnnotateTokensData::iterator Pos = Annotated.find(Tokens[I].int_data[1]);
    Cursors[I] = (Pos == Annotated.end()) ? C : Pos->second;
  }
}

enum CXChildVisitResult
AnnotateTokensWorker::Visit(CXCursor cursor, CXCursor parent) {  
  CXSourceLocation Loc = clang_getCursorLocation(cursor);
  SourceRange cursorRange = getRawCursorExtent(cursor);
  if (cursorRange.isInvalid())
    return CXChildVisit_Recurse;
        
  if (clang_isPreprocessing(cursor.kind)) {    
    // For macro instantiations, just note where the beginning of the macro
    // instantiation occurs.
    if (cursor.kind == CXCursor_MacroInstantiation) {
      Annotated[Loc.int_data] = cursor;
      return CXChildVisit_Recurse;
    }
    
    // Items in the preprocessing record are kept separate from items in
    // declarations, so we keep a separate token index.
    unsigned SavedTokIdx = TokIdx;
    TokIdx = PreprocessingTokIdx;

    // Skip tokens up until we catch up to the beginning of the preprocessing
    // entry.
    while (MoreTokens()) {
      const unsigned I = NextToken();
      SourceLocation TokLoc = GetTokenLoc(I);
      switch (LocationCompare(SrcMgr, TokLoc, cursorRange)) {
      case RangeBefore:
        AdvanceToken();
        continue;
      case RangeAfter:
      case RangeOverlap:
        break;
      }
      break;
    }
    
    // Look at all of the tokens within this range.
    while (MoreTokens()) {
      const unsigned I = NextToken();
      SourceLocation TokLoc = GetTokenLoc(I);
      switch (LocationCompare(SrcMgr, TokLoc, cursorRange)) {
      case RangeBefore:
        assert(0 && "Infeasible");
      case RangeAfter:
        break;
      case RangeOverlap:
        Cursors[I] = cursor;
        AdvanceToken();
        continue;
      }
      break;
    }

    // Save the preprocessing token index; restore the non-preprocessing
    // token index.
    PreprocessingTokIdx = TokIdx;
    TokIdx = SavedTokIdx;
    return CXChildVisit_Recurse;
  }

  if (cursorRange.isInvalid())
    return CXChildVisit_Continue;
  
  SourceLocation L = SourceLocation::getFromRawEncoding(Loc.int_data);

  // Adjust the annotated range based specific declarations.
  const enum CXCursorKind cursorK = clang_getCursorKind(cursor);
  if (cursorK >= CXCursor_FirstDecl && cursorK <= CXCursor_LastDecl) {
    Decl *D = cxcursor::getCursorDecl(cursor);
    // Don't visit synthesized ObjC methods, since they have no syntatic
    // representation in the source.
    if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
      if (MD->isSynthesized())
        return CXChildVisit_Continue;
    }
    
    SourceLocation StartLoc;
    if (const DeclaratorDecl *DD = dyn_cast<DeclaratorDecl>(D)) {
      if (TypeSourceInfo *TI = DD->getTypeSourceInfo())
        StartLoc = TI->getTypeLoc().getSourceRange().getBegin();
    } else if (TypedefDecl *Typedef = dyn_cast<TypedefDecl>(D)) {
      if (TypeSourceInfo *TI = Typedef->getTypeSourceInfo())
        StartLoc = TI->getTypeLoc().getSourceRange().getBegin();
    }

    if (StartLoc.isValid() && L.isValid() &&
        SrcMgr.isBeforeInTranslationUnit(StartLoc, L))
      cursorRange.setBegin(StartLoc);
  }
  
  // If the location of the cursor occurs within a macro instantiation, record
  // the spelling location of the cursor in our annotation map.  We can then
  // paper over the token labelings during a post-processing step to try and
  // get cursor mappings for tokens that are the *arguments* of a macro
  // instantiation.
  if (L.isMacroID()) {
    unsigned rawEncoding = SrcMgr.getSpellingLoc(L).getRawEncoding();
    // Only invalidate the old annotation if it isn't part of a preprocessing
    // directive.  Here we assume that the default construction of CXCursor
    // results in CXCursor.kind being an initialized value (i.e., 0).  If
    // this isn't the case, we can fix by doing lookup + insertion.
    
    CXCursor &oldC = Annotated[rawEncoding];
    if (!clang_isPreprocessing(oldC.kind))
      oldC = cursor;
  }
  
  const enum CXCursorKind K = clang_getCursorKind(parent);
  const CXCursor updateC =
    (clang_isInvalid(K) || K == CXCursor_TranslationUnit)
     ? clang_getNullCursor() : parent;

  while (MoreTokens()) {
    const unsigned I = NextToken();
    SourceLocation TokLoc = GetTokenLoc(I);
    switch (LocationCompare(SrcMgr, TokLoc, cursorRange)) {
      case RangeBefore:
        Cursors[I] = updateC;
        AdvanceToken();
        continue;
      case RangeAfter:
      case RangeOverlap:
        break;
    }
    break;
  }

  // Visit children to get their cursor information.
  const unsigned BeforeChildren = NextToken();
  VisitChildren(cursor);
  const unsigned AfterChildren = NextToken();

  // Adjust 'Last' to the last token within the extent of the cursor.
  while (MoreTokens()) {
    const unsigned I = NextToken();
    SourceLocation TokLoc = GetTokenLoc(I);
    switch (LocationCompare(SrcMgr, TokLoc, cursorRange)) {
      case RangeBefore:
        assert(0 && "Infeasible");
      case RangeAfter:
        break;
      case RangeOverlap:
        Cursors[I] = updateC;
        AdvanceToken();
        continue;
    }
    break;
  }
  const unsigned Last = NextToken();
  
  // Scan the tokens that are at the beginning of the cursor, but are not
  // capture by the child cursors.

  // For AST elements within macros, rely on a post-annotate pass to
  // to correctly annotate the tokens with cursors.  Otherwise we can
  // get confusing results of having tokens that map to cursors that really
  // are expanded by an instantiation.
  if (L.isMacroID())
    cursor = clang_getNullCursor();

  for (unsigned I = BeforeChildren; I != AfterChildren; ++I) {
    if (!clang_isInvalid(clang_getCursorKind(Cursors[I])))
      break;
    
    Cursors[I] = cursor;
  }
  // Scan the tokens that are at the end of the cursor, but are not captured
  // but the child cursors.
  for (unsigned I = AfterChildren; I != Last; ++I)
    Cursors[I] = cursor;

  TokIdx = Last;
  return CXChildVisit_Continue;
}

static enum CXChildVisitResult AnnotateTokensVisitor(CXCursor cursor,
                                                     CXCursor parent,
                                                     CXClientData client_data) {
  return static_cast<AnnotateTokensWorker*>(client_data)->Visit(cursor, parent);
}

// This gets run a separate thread to avoid stack blowout.
static void runAnnotateTokensWorker(void *UserData) {
  ((AnnotateTokensWorker*)UserData)->AnnotateTokens();
}

extern "C" {

void clang_annotateTokens(CXTranslationUnit TU,
                          CXToken *Tokens, unsigned NumTokens,
                          CXCursor *Cursors) {

  if (NumTokens == 0 || !Tokens || !Cursors)
    return;

  // Any token we don't specifically annotate will have a NULL cursor.
  CXCursor C = clang_getNullCursor();
  for (unsigned I = 0; I != NumTokens; ++I)
    Cursors[I] = C;

  ASTUnit *CXXUnit = static_cast<ASTUnit *>(TU->TUData);
  if (!CXXUnit)
    return;

  ASTUnit::ConcurrencyCheck Check(*CXXUnit);

  // Determine the region of interest, which contains all of the tokens.
  SourceRange RegionOfInterest;
  RegionOfInterest.setBegin(cxloc::translateSourceLocation(
                                        clang_getTokenLocation(TU, Tokens[0])));
  RegionOfInterest.setEnd(cxloc::translateSourceLocation(
                                clang_getTokenLocation(TU, 
                                                       Tokens[NumTokens - 1])));

  // A mapping from the source locations found when re-lexing or traversing the
  // region of interest to the corresponding cursors.
  AnnotateTokensData Annotated;

  // Relex the tokens within the source range to look for preprocessing
  // directives.
  SourceManager &SourceMgr = CXXUnit->getSourceManager();
  std::pair<FileID, unsigned> BeginLocInfo
    = SourceMgr.getDecomposedLoc(RegionOfInterest.getBegin());
  std::pair<FileID, unsigned> EndLocInfo
    = SourceMgr.getDecomposedLoc(RegionOfInterest.getEnd());

  llvm::StringRef Buffer;
  bool Invalid = false;
  if (BeginLocInfo.first == EndLocInfo.first &&
      ((Buffer = SourceMgr.getBufferData(BeginLocInfo.first, &Invalid)),true) &&
      !Invalid) {
    Lexer Lex(SourceMgr.getLocForStartOfFile(BeginLocInfo.first),
              CXXUnit->getASTContext().getLangOptions(),
              Buffer.begin(), Buffer.data() + BeginLocInfo.second,
              Buffer.end());
    Lex.SetCommentRetentionState(true);

    // Lex tokens in raw mode until we hit the end of the range, to avoid
    // entering #includes or expanding macros.
    while (true) {
      Token Tok;
      Lex.LexFromRawLexer(Tok);

    reprocess:
      if (Tok.is(tok::hash) && Tok.isAtStartOfLine()) {
        // We have found a preprocessing directive. Gobble it up so that we
        // don't see it while preprocessing these tokens later, but keep track
        // of all of the token locations inside this preprocessing directive so
        // that we can annotate them appropriately.
        //
        // FIXME: Some simple tests here could identify macro definitions and
        // #undefs, to provide specific cursor kinds for those.
        std::vector<SourceLocation> Locations;
        do {
          Locations.push_back(Tok.getLocation());
          Lex.LexFromRawLexer(Tok);
        } while (!Tok.isAtStartOfLine() && !Tok.is(tok::eof));

        using namespace cxcursor;
        CXCursor Cursor
          = MakePreprocessingDirectiveCursor(SourceRange(Locations.front(),
                                                         Locations.back()),
                                           TU);
        for (unsigned I = 0, N = Locations.size(); I != N; ++I) {
          Annotated[Locations[I].getRawEncoding()] = Cursor;
        }

        if (Tok.isAtStartOfLine())
          goto reprocess;

        continue;
      }

      if (Tok.is(tok::eof))
        break;
    }
  }

  // Annotate all of the source locations in the region of interest that map to
  // a specific cursor.
  AnnotateTokensWorker W(Annotated, Tokens, Cursors, NumTokens,
                         TU, RegionOfInterest);

  // Run the worker within a CrashRecoveryContext.
  // FIXME: We use a ridiculous stack size here because the data-recursion
  // algorithm uses a large stack frame than the non-data recursive version,
  // and AnnotationTokensWorker currently transforms the data-recursion
  // algorithm back into a traditional recursion by explicitly calling
  // VisitChildren().  We will need to remove this explicit recursive call.
  llvm::CrashRecoveryContext CRC;
  if (!RunSafely(CRC, runAnnotateTokensWorker, &W,
                 GetSafetyThreadStackSize() * 2)) {
    fprintf(stderr, "libclang: crash detected while annotating tokens\n");
  }
}
} // end: extern "C"

//===----------------------------------------------------------------------===//
// Operations for querying linkage of a cursor.
//===----------------------------------------------------------------------===//

extern "C" {
CXLinkageKind clang_getCursorLinkage(CXCursor cursor) {
  if (!clang_isDeclaration(cursor.kind))
    return CXLinkage_Invalid;

  Decl *D = cxcursor::getCursorDecl(cursor);
  if (NamedDecl *ND = dyn_cast_or_null<NamedDecl>(D))
    switch (ND->getLinkage()) {
      case NoLinkage: return CXLinkage_NoLinkage;
      case InternalLinkage: return CXLinkage_Internal;
      case UniqueExternalLinkage: return CXLinkage_UniqueExternal;
      case ExternalLinkage: return CXLinkage_External;
    };

  return CXLinkage_Invalid;
}
} // end: extern "C"

//===----------------------------------------------------------------------===//
// Operations for querying language of a cursor.
//===----------------------------------------------------------------------===//

static CXLanguageKind getDeclLanguage(const Decl *D) {
  switch (D->getKind()) {
    default:
      break;
    case Decl::ImplicitParam:
    case Decl::ObjCAtDefsField:
    case Decl::ObjCCategory:
    case Decl::ObjCCategoryImpl:
    case Decl::ObjCClass:
    case Decl::ObjCCompatibleAlias:
    case Decl::ObjCForwardProtocol:
    case Decl::ObjCImplementation:
    case Decl::ObjCInterface:
    case Decl::ObjCIvar:
    case Decl::ObjCMethod:
    case Decl::ObjCProperty:
    case Decl::ObjCPropertyImpl:
    case Decl::ObjCProtocol:
      return CXLanguage_ObjC;
    case Decl::CXXConstructor:
    case Decl::CXXConversion:
    case Decl::CXXDestructor:
    case Decl::CXXMethod:
    case Decl::CXXRecord:
    case Decl::ClassTemplate:
    case Decl::ClassTemplatePartialSpecialization:
    case Decl::ClassTemplateSpecialization:
    case Decl::Friend:
    case Decl::FriendTemplate:
    case Decl::FunctionTemplate:
    case Decl::LinkageSpec:
    case Decl::Namespace:
    case Decl::NamespaceAlias:
    case Decl::NonTypeTemplateParm:
    case Decl::StaticAssert:
    case Decl::TemplateTemplateParm:
    case Decl::TemplateTypeParm:
    case Decl::UnresolvedUsingTypename:
    case Decl::UnresolvedUsingValue:
    case Decl::Using:
    case Decl::UsingDirective:
    case Decl::UsingShadow:
      return CXLanguage_CPlusPlus;
  }

  return CXLanguage_C;
}

extern "C" {
  
enum CXAvailabilityKind clang_getCursorAvailability(CXCursor cursor) {
  if (clang_isDeclaration(cursor.kind))
    if (Decl *D = cxcursor::getCursorDecl(cursor)) {
      if (D->hasAttr<UnavailableAttr>() ||
          (isa<FunctionDecl>(D) && cast<FunctionDecl>(D)->isDeleted()))
        return CXAvailability_Available;
      
      if (D->hasAttr<DeprecatedAttr>())
        return CXAvailability_Deprecated;
    }
  
  return CXAvailability_Available;
}

CXLanguageKind clang_getCursorLanguage(CXCursor cursor) {
  if (clang_isDeclaration(cursor.kind))
    return getDeclLanguage(cxcursor::getCursorDecl(cursor));

  return CXLanguage_Invalid;
}

 /// \brief If the given cursor is the "templated" declaration
 /// descibing a class or function template, return the class or
 /// function template.
static Decl *maybeGetTemplateCursor(Decl *D) {
  if (!D)
    return 0;

  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    if (FunctionTemplateDecl *FunTmpl = FD->getDescribedFunctionTemplate())
      return FunTmpl;

  if (CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D))
    if (ClassTemplateDecl *ClassTmpl = RD->getDescribedClassTemplate())
      return ClassTmpl;

  return D;
}

CXCursor clang_getCursorSemanticParent(CXCursor cursor) {
  if (clang_isDeclaration(cursor.kind)) {
    if (Decl *D = getCursorDecl(cursor)) {
      DeclContext *DC = D->getDeclContext();
      if (!DC)
        return clang_getNullCursor();

      return MakeCXCursor(maybeGetTemplateCursor(cast<Decl>(DC)), 
                          getCursorTU(cursor));
    }
  }
  
  if (clang_isStatement(cursor.kind) || clang_isExpression(cursor.kind)) {
    if (Decl *D = getCursorDecl(cursor))
      return MakeCXCursor(D, getCursorTU(cursor));
  }
  
  return clang_getNullCursor();
}

CXCursor clang_getCursorLexicalParent(CXCursor cursor) {
  if (clang_isDeclaration(cursor.kind)) {
    if (Decl *D = getCursorDecl(cursor)) {
      DeclContext *DC = D->getLexicalDeclContext();
      if (!DC)
        return clang_getNullCursor();

      return MakeCXCursor(maybeGetTemplateCursor(cast<Decl>(DC)), 
                          getCursorTU(cursor));
    }
  }

  // FIXME: Note that we can't easily compute the lexical context of a 
  // statement or expression, so we return nothing.
  return clang_getNullCursor();
}

static void CollectOverriddenMethods(DeclContext *Ctx, 
                                     ObjCMethodDecl *Method,
                            llvm::SmallVectorImpl<ObjCMethodDecl *> &Methods) {
  if (!Ctx)
    return;

  // If we have a class or category implementation, jump straight to the 
  // interface.
  if (ObjCImplDecl *Impl = dyn_cast<ObjCImplDecl>(Ctx))
    return CollectOverriddenMethods(Impl->getClassInterface(), Method, Methods);
  
  ObjCContainerDecl *Container = dyn_cast<ObjCContainerDecl>(Ctx);
  if (!Container)
    return;

  // Check whether we have a matching method at this level.
  if (ObjCMethodDecl *Overridden = Container->getMethod(Method->getSelector(),
                                                    Method->isInstanceMethod()))
    if (Method != Overridden) {
      // We found an override at this level; there is no need to look
      // into other protocols or categories.
      Methods.push_back(Overridden);
      return;
    }

  if (ObjCProtocolDecl *Protocol = dyn_cast<ObjCProtocolDecl>(Container)) {
    for (ObjCProtocolDecl::protocol_iterator P = Protocol->protocol_begin(),
                                          PEnd = Protocol->protocol_end();
         P != PEnd; ++P)
      CollectOverriddenMethods(*P, Method, Methods);
  }

  if (ObjCCategoryDecl *Category = dyn_cast<ObjCCategoryDecl>(Container)) {
    for (ObjCCategoryDecl::protocol_iterator P = Category->protocol_begin(),
                                          PEnd = Category->protocol_end();
         P != PEnd; ++P)
      CollectOverriddenMethods(*P, Method, Methods);
  }

  if (ObjCInterfaceDecl *Interface = dyn_cast<ObjCInterfaceDecl>(Container)) {
    for (ObjCInterfaceDecl::protocol_iterator P = Interface->protocol_begin(),
                                           PEnd = Interface->protocol_end();
         P != PEnd; ++P)
      CollectOverriddenMethods(*P, Method, Methods);

    for (ObjCCategoryDecl *Category = Interface->getCategoryList();
         Category; Category = Category->getNextClassCategory())
      CollectOverriddenMethods(Category, Method, Methods);

    // We only look into the superclass if we haven't found anything yet.
    if (Methods.empty())
      if (ObjCInterfaceDecl *Super = Interface->getSuperClass())
        return CollectOverriddenMethods(Super, Method, Methods);
  }
}

void clang_getOverriddenCursors(CXCursor cursor, 
                                CXCursor **overridden,
                                unsigned *num_overridden) {
  if (overridden)
    *overridden = 0;
  if (num_overridden)
    *num_overridden = 0;
  if (!overridden || !num_overridden)
    return;

  if (!clang_isDeclaration(cursor.kind))
    return;

  Decl *D = getCursorDecl(cursor);
  if (!D)
    return;

  // Handle C++ member functions.
  CXTranslationUnit TU = getCursorTU(cursor);
  if (CXXMethodDecl *CXXMethod = dyn_cast<CXXMethodDecl>(D)) {
    *num_overridden = CXXMethod->size_overridden_methods();
    if (!*num_overridden)
      return;

    *overridden = new CXCursor [*num_overridden];
    unsigned I = 0;
    for (CXXMethodDecl::method_iterator
              M = CXXMethod->begin_overridden_methods(),
           MEnd = CXXMethod->end_overridden_methods();
         M != MEnd; (void)++M, ++I)
      (*overridden)[I] = MakeCXCursor(const_cast<CXXMethodDecl*>(*M), TU);
    return;
  }

  ObjCMethodDecl *Method = dyn_cast<ObjCMethodDecl>(D);
  if (!Method)
    return;

  // Handle Objective-C methods.
  llvm::SmallVector<ObjCMethodDecl *, 4> Methods;
  CollectOverriddenMethods(Method->getDeclContext(), Method, Methods);

  if (Methods.empty())
    return;

  *num_overridden = Methods.size();
  *overridden = new CXCursor [Methods.size()];
  for (unsigned I = 0, N = Methods.size(); I != N; ++I)
    (*overridden)[I] = MakeCXCursor(Methods[I], TU); 
}

void clang_disposeOverriddenCursors(CXCursor *overridden) {
  delete [] overridden;
}

CXFile clang_getIncludedFile(CXCursor cursor) {
  if (cursor.kind != CXCursor_InclusionDirective)
    return 0;
  
  InclusionDirective *ID = getCursorInclusionDirective(cursor);
  return (void *)ID->getFile();
}
  
} // end: extern "C"


//===----------------------------------------------------------------------===//
// C++ AST instrospection.
//===----------------------------------------------------------------------===//

extern "C" {
unsigned clang_CXXMethod_isStatic(CXCursor C) {
  if (!clang_isDeclaration(C.kind))
    return 0;
  
  CXXMethodDecl *Method = 0;
  Decl *D = cxcursor::getCursorDecl(C);
  if (FunctionTemplateDecl *FunTmpl = dyn_cast_or_null<FunctionTemplateDecl>(D))
    Method = dyn_cast<CXXMethodDecl>(FunTmpl->getTemplatedDecl());
  else
    Method = dyn_cast_or_null<CXXMethodDecl>(D);
  return (Method && Method->isStatic()) ? 1 : 0;
}

} // end: extern "C"

//===----------------------------------------------------------------------===//
// Attribute introspection.
//===----------------------------------------------------------------------===//

extern "C" {
CXType clang_getIBOutletCollectionType(CXCursor C) {
  if (C.kind != CXCursor_IBOutletCollectionAttr)
    return cxtype::MakeCXType(QualType(), cxcursor::getCursorTU(C));
  
  IBOutletCollectionAttr *A =
    cast<IBOutletCollectionAttr>(cxcursor::getCursorAttr(C));
  
  return cxtype::MakeCXType(A->getInterFace(), cxcursor::getCursorTU(C));  
}
} // end: extern "C"

//===----------------------------------------------------------------------===//
// Misc. utility functions.
//===----------------------------------------------------------------------===//

/// Default to using an 8 MB stack size on "safety" threads.
static unsigned SafetyStackThreadSize = 8 << 20;

namespace clang {

bool RunSafely(llvm::CrashRecoveryContext &CRC,
               void (*Fn)(void*), void *UserData,
               unsigned Size) {
  if (!Size)
    Size = GetSafetyThreadStackSize();
  if (Size)
    return CRC.RunSafelyOnThread(Fn, UserData, Size);
  return CRC.RunSafely(Fn, UserData);
}

unsigned GetSafetyThreadStackSize() {
  return SafetyStackThreadSize;
}

void SetSafetyThreadStackSize(unsigned Value) {
  SafetyStackThreadSize = Value;
}

}

extern "C" {

CXString clang_getClangVersion() {
  return createCXString(getClangFullVersion());
}

} // end: extern "C"
