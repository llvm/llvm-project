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
#include "CXComment.h"
#include "CXCursor.h"
#include "CXTranslationUnit.h"
#include "CXString.h"
#include "CXType.h"
#include "CXSourceLocation.h"
#include "CIndexDiagnostic.h"
#include "CursorVisitor.h"

#include "clang/Basic/Version.h"

#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/SaveAndRestore.h"
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
using namespace clang::cxtu;
using namespace clang::cxindex;

CXTranslationUnit cxtu::MakeCXTranslationUnit(CIndexer *CIdx, ASTUnit *TU) {
  if (!TU)
    return 0;
  CXTranslationUnit D = new CXTranslationUnitImpl();
  D->CIdx = CIdx;
  D->TUData = TU;
  D->StringPool = createCXStringPool();
  D->Diagnostics = 0;
  D->OverridenCursorsPool = createOverridenCXCursorsPool();
  return D;
}

cxtu::CXTUOwner::~CXTUOwner() {
  if (TU)
    clang_disposeTranslationUnit(TU);
}

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
  if (EndLoc.isValid() && EndLoc.isMacroID() && !SM.isMacroArgExpansion(EndLoc))
    EndLoc = SM.getExpansionRange(EndLoc).second;
  if (R.isTokenRange() && !EndLoc.isInvalid()) {
    unsigned Length = Lexer::MeasureTokenLength(SM.getSpellingLoc(EndLoc),
                                                SM, LangOpts);
    EndLoc = EndLoc.getLocWithOffset(Length);
  }

  CXSourceRange Result = { { (void *)&SM, (void *)&LangOpts },
                           R.getBegin().getRawEncoding(),
                           EndLoc.getRawEncoding() };
  return Result;
}

//===----------------------------------------------------------------------===//
// Cursor visitor.
//===----------------------------------------------------------------------===//

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
/// \param CheckedRegionOfInterest if true, then the caller already checked
/// that this cursor is within the region of interest.
///
/// \returns true if the visitation should be aborted, false if it
/// should continue.
bool CursorVisitor::Visit(CXCursor Cursor, bool CheckedRegionOfInterest) {
  if (clang_isInvalid(Cursor.kind))
    return false;

  if (clang_isDeclaration(Cursor.kind)) {
    Decl *D = getCursorDecl(Cursor);
    if (!D) {
      assert(0 && "Invalid declaration cursor");
      return true; // abort.
    }
    
    // Ignore implicit declarations, unless it's an objc method because
    // currently we should report implicit methods for properties when indexing.
    if (D->isImplicit() && !isa<ObjCMethodDecl>(D))
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

  case CXChildVisit_Recurse: {
    bool ret = VisitChildren(Cursor);
    if (PostChildrenVisitor)
      if (PostChildrenVisitor(Cursor, ClientData))
        return true;
    return ret;
  }
  }

  llvm_unreachable("Invalid CXChildVisitResult!");
}

static bool visitPreprocessedEntitiesInRange(SourceRange R,
                                             PreprocessingRecord &PPRec,
                                             CursorVisitor &Visitor) {
  SourceManager &SM = Visitor.getASTUnit()->getSourceManager();
  FileID FID;
  
  if (!Visitor.shouldVisitIncludedEntities()) {
    // If the begin/end of the range lie in the same FileID, do the optimization
    // where we skip preprocessed entities that do not come from the same FileID.
    FID = SM.getFileID(SM.getFileLoc(R.getBegin()));
    if (FID != SM.getFileID(SM.getFileLoc(R.getEnd())))
      FID = FileID();
  }

  std::pair<PreprocessingRecord::iterator, PreprocessingRecord::iterator>
    Entities = PPRec.getPreprocessedEntitiesInRange(R);
  return Visitor.visitPreprocessedEntities(Entities.first, Entities.second,
                                           PPRec, FID);
}

void CursorVisitor::visitFileRegion() {
  if (RegionOfInterest.isInvalid())
    return;

  ASTUnit *Unit = static_cast<ASTUnit *>(TU->TUData);
  SourceManager &SM = Unit->getSourceManager();
  
  std::pair<FileID, unsigned>
    Begin = SM.getDecomposedLoc(SM.getFileLoc(RegionOfInterest.getBegin())), 
    End = SM.getDecomposedLoc(SM.getFileLoc(RegionOfInterest.getEnd())); 

  if (End.first != Begin.first) {
    // If the end does not reside in the same file, try to recover by
    // picking the end of the file of begin location.
    End.first = Begin.first;
    End.second = SM.getFileIDSize(Begin.first);
  }

  assert(Begin.first == End.first);
  if (Begin.second > End.second)
    return;
  
  FileID File = Begin.first;
  unsigned Offset = Begin.second;
  unsigned Length = End.second - Begin.second;

  if (!VisitDeclsOnly && !VisitPreprocessorLast)
    if (visitPreprocessedEntitiesInRegion())
      return; // visitation break.

  visitDeclsFromFileRegion(File, Offset, Length);

  if (!VisitDeclsOnly && VisitPreprocessorLast)
    visitPreprocessedEntitiesInRegion();
}

static bool isInLexicalContext(Decl *D, DeclContext *DC) {
  if (!DC)
    return false;

  for (DeclContext *DeclDC = D->getLexicalDeclContext();
         DeclDC; DeclDC = DeclDC->getLexicalParent()) {
    if (DeclDC == DC)
      return true;
  }
  return false;
}

void CursorVisitor::visitDeclsFromFileRegion(FileID File,
                                             unsigned Offset, unsigned Length) {
  ASTUnit *Unit = static_cast<ASTUnit *>(TU->TUData);
  SourceManager &SM = Unit->getSourceManager();
  SourceRange Range = RegionOfInterest;

  SmallVector<Decl *, 16> Decls;
  Unit->findFileRegionDecls(File, Offset, Length, Decls);

  // If we didn't find any file level decls for the file, try looking at the
  // file that it was included from.
  while (Decls.empty() || Decls.front()->isTopLevelDeclInObjCContainer()) {
    bool Invalid = false;
    const SrcMgr::SLocEntry &SLEntry = SM.getSLocEntry(File, &Invalid);
    if (Invalid)
      return;

    SourceLocation Outer;
    if (SLEntry.isFile())
      Outer = SLEntry.getFile().getIncludeLoc();
    else
      Outer = SLEntry.getExpansion().getExpansionLocStart();
    if (Outer.isInvalid())
      return;

    llvm::tie(File, Offset) = SM.getDecomposedExpansionLoc(Outer);
    Length = 0;
    Unit->findFileRegionDecls(File, Offset, Length, Decls);
  }

  assert(!Decls.empty());

  bool VisitedAtLeastOnce = false;
  DeclContext *CurDC = 0;
  SmallVector<Decl *, 16>::iterator DIt = Decls.begin();
  for (SmallVector<Decl *, 16>::iterator DE = Decls.end(); DIt != DE; ++DIt) {
    Decl *D = *DIt;
    if (D->getSourceRange().isInvalid())
      continue;

    if (isInLexicalContext(D, CurDC))
      continue;

    CurDC = dyn_cast<DeclContext>(D);

    if (TagDecl *TD = dyn_cast<TagDecl>(D))
      if (!TD->isFreeStanding())
        continue;

    RangeComparisonResult CompRes = RangeCompare(SM, D->getSourceRange(),Range);
    if (CompRes == RangeBefore)
      continue;
    if (CompRes == RangeAfter)
      break;

    assert(CompRes == RangeOverlap);
    VisitedAtLeastOnce = true;

    if (isa<ObjCContainerDecl>(D)) {
      FileDI_current = &DIt;
      FileDE_current = DE;
    } else {
      FileDI_current = 0;
    }

    if (Visit(MakeCXCursor(D, TU, Range), /*CheckedRegionOfInterest=*/true))
      break;
  }

  if (VisitedAtLeastOnce)
    return;

  // No Decls overlapped with the range. Move up the lexical context until there
  // is a context that contains the range or we reach the translation unit
  // level.
  DeclContext *DC = DIt == Decls.begin() ? (*DIt)->getLexicalDeclContext()
                                         : (*(DIt-1))->getLexicalDeclContext();

  while (DC && !DC->isTranslationUnit()) {
    Decl *D = cast<Decl>(DC);
    SourceRange CurDeclRange = D->getSourceRange();
    if (CurDeclRange.isInvalid())
      break;

    if (RangeCompare(SM, CurDeclRange, Range) == RangeOverlap) {
      Visit(MakeCXCursor(D, TU, Range), /*CheckedRegionOfInterest=*/true);
      break;
    }

    DC = D->getLexicalDeclContext();
  }
}

bool CursorVisitor::visitPreprocessedEntitiesInRegion() {
  if (!AU->getPreprocessor().getPreprocessingRecord())
    return false;

  PreprocessingRecord &PPRec
    = *AU->getPreprocessor().getPreprocessingRecord();
  SourceManager &SM = AU->getSourceManager();
  
  if (RegionOfInterest.isValid()) {
    SourceRange MappedRange = AU->mapRangeToPreamble(RegionOfInterest);
    SourceLocation B = MappedRange.getBegin();
    SourceLocation E = MappedRange.getEnd();

    if (AU->isInPreambleFileID(B)) {
      if (SM.isLoadedSourceLocation(E))
        return visitPreprocessedEntitiesInRange(SourceRange(B, E),
                                                 PPRec, *this);

      // Beginning of range lies in the preamble but it also extends beyond
      // it into the main file. Split the range into 2 parts, one covering
      // the preamble and another covering the main file. This allows subsequent
      // calls to visitPreprocessedEntitiesInRange to accept a source range that
      // lies in the same FileID, allowing it to skip preprocessed entities that
      // do not come from the same FileID.
      bool breaked =
        visitPreprocessedEntitiesInRange(
                                   SourceRange(B, AU->getEndOfPreambleFileID()),
                                          PPRec, *this);
      if (breaked) return true;
      return visitPreprocessedEntitiesInRange(
                                    SourceRange(AU->getStartOfMainFileID(), E),
                                        PPRec, *this);
    }

    return visitPreprocessedEntitiesInRange(SourceRange(B, E), PPRec, *this);
  }

  bool OnlyLocalDecls
    = !AU->isMainFileAST() && AU->getOnlyLocalDecls(); 
  
  if (OnlyLocalDecls)
    return visitPreprocessedEntities(PPRec.local_begin(), PPRec.local_end(),
                                     PPRec);

  return visitPreprocessedEntities(PPRec.begin(), PPRec.end(), PPRec);
}

template<typename InputIterator>
bool CursorVisitor::visitPreprocessedEntities(InputIterator First,
                                              InputIterator Last,
                                              PreprocessingRecord &PPRec,
                                              FileID FID) {
  for (; First != Last; ++First) {
    if (!FID.isInvalid() && !PPRec.isEntityInFileID(First, FID))
      continue;

    PreprocessedEntity *PPE = *First;
    if (MacroExpansion *ME = dyn_cast<MacroExpansion>(PPE)) {
      if (Visit(MakeMacroExpansionCursor(ME, TU)))
        return true;
      
      continue;
    }
    
    if (MacroDefinition *MD = dyn_cast<MacroDefinition>(PPE)) {
      if (Visit(MakeMacroDefinitionCursor(MD, TU)))
        return true;
      
      continue;
    }
    
    if (InclusionDirective *ID = dyn_cast<InclusionDirective>(PPE)) {
      if (Visit(MakeInclusionDirectiveCursor(ID, TU)))
        return true;
      
      continue;
    }
  }

  return false;
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
    if (!D)
      return false;

    return VisitAttributes(D) || Visit(D);
  }

  if (clang_isStatement(Cursor.kind)) {
    if (Stmt *S = getCursorStmt(Cursor))
      return Visit(S);

    return false;
  }

  if (clang_isExpression(Cursor.kind)) {
    if (Expr *E = getCursorExpr(Cursor))
      return Visit(E);

    return false;
  }

  if (clang_isTranslationUnit(Cursor.kind)) {
    CXTranslationUnit tu = getCursorTU(Cursor);
    ASTUnit *CXXUnit = static_cast<ASTUnit*>(tu->TUData);
    
    int VisitOrder[2] = { VisitPreprocessorLast, !VisitPreprocessorLast };
    for (unsigned I = 0; I != 2; ++I) {
      if (VisitOrder[I]) {
        if (!CXXUnit->isMainFileAST() && CXXUnit->getOnlyLocalDecls() &&
            RegionOfInterest.isInvalid()) {
          for (ASTUnit::top_level_iterator TL = CXXUnit->top_level_begin(),
                                        TLEnd = CXXUnit->top_level_end();
               TL != TLEnd; ++TL) {
            if (Visit(MakeCXCursor(*TL, tu, RegionOfInterest), true))
              return true;
          }
        } else if (VisitDeclContext(
                                CXXUnit->getASTContext().getTranslationUnitDecl()))
          return true;
        continue;
      }

      // Walk the preprocessing record.
      if (CXXUnit->getPreprocessor().getPreprocessingRecord())
        visitPreprocessedEntitiesInRegion();
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

  if (Cursor.kind == CXCursor_IBOutletCollectionAttr) {
    IBOutletCollectionAttr *A =
      cast<IBOutletCollectionAttr>(cxcursor::getCursorAttr(Cursor));
    if (const ObjCInterfaceType *InterT = A->getInterface()->getAs<ObjCInterfaceType>())
      return Visit(cxcursor::MakeCursorObjCClassRef(InterT->getInterface(),
                                                    A->getInterfaceLoc(), TU));
  }

  // Nothing to visit at the moment.
  return false;
}

bool CursorVisitor::VisitBlockDecl(BlockDecl *B) {
  if (TypeSourceInfo *TSInfo = B->getSignatureAsWritten())
    if (Visit(TSInfo->getTypeLoc()))
        return true;

  if (Stmt *Body = B->getBody())
    return Visit(MakeCXCursor(Body, StmtParent, TU, RegionOfInterest));

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
    CXCursor Cursor = MakeCXCursor(D, TU, RegionOfInterest);

    // Ignore synthesized ivars here, otherwise if we have something like:
    //   @synthesize prop = _prop;
    // and '_prop' is not declared, we will encounter a '_prop' ivar before
    // encountering the 'prop' synthesize declaration and we will think that
    // we passed the region-of-interest.
    if (ObjCIvarDecl *ivarD = dyn_cast<ObjCIvarDecl>(D)) {
      if (ivarD->getSynthesize())
        continue;
    }

    // FIXME: ObjCClassRef/ObjCProtocolRef for forward class/protocol
    // declarations is a mismatch with the compiler semantics.
    if (Cursor.kind == CXCursor_ObjCInterfaceDecl) {
      ObjCInterfaceDecl *ID = cast<ObjCInterfaceDecl>(D);
      if (!ID->isThisDeclarationADefinition())
        Cursor = MakeCursorObjCClassRef(ID, ID->getLocation(), TU);

    } else if (Cursor.kind == CXCursor_ObjCProtocolDecl) {
      ObjCProtocolDecl *PD = cast<ObjCProtocolDecl>(D);
      if (!PD->isThisDeclarationADefinition())
        Cursor = MakeCursorObjCProtocolRef(PD, PD->getLocation(), TU);
    }

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
}

bool CursorVisitor::VisitTypeAliasDecl(TypeAliasDecl *D) {
  if (TypeSourceInfo *TSInfo = D->getTypeSourceInfo())
    return Visit(TSInfo->getTypeLoc());

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
    return Visit(MakeCXCursor(Init, StmtParent, TU, RegionOfInterest));
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
  
  if (ND->doesThisDeclarationHaveABody() && !ND->isLateTemplateParsed()) {
    if (CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(ND)) {
      // Find the initializers that were written in the source.
      SmallVector<CXXCtorInitializer *, 4> WrittenInits;
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
        } else if (TypeSourceInfo *TInfo = Init->getTypeSourceInfo()) {
          if (Visit(TInfo->getTypeLoc()))
            return true;
        }
        
        // Visit the initializer value.
        if (Expr *Initializer = Init->getInit())
          if (Visit(MakeCXCursor(Initializer, ND, TU, RegionOfInterest)))
            return true;
      } 
    }
    
    if (Visit(MakeCXCursor(ND->getBody(), StmtParent, TU, RegionOfInterest)))
      return true;
  }

  return false;
}

bool CursorVisitor::VisitFieldDecl(FieldDecl *D) {
  if (VisitDeclaratorDecl(D))
    return true;

  if (Expr *BitWidth = D->getBitWidth())
    return Visit(MakeCXCursor(BitWidth, StmtParent, TU, RegionOfInterest));

  return false;
}

bool CursorVisitor::VisitVarDecl(VarDecl *D) {
  if (VisitDeclaratorDecl(D))
    return true;

  if (Expr *Init = D->getInit())
    return Visit(MakeCXCursor(Init, StmtParent, TU, RegionOfInterest));

  return false;
}

bool CursorVisitor::VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D) {
  if (VisitDeclaratorDecl(D))
    return true;
  
  if (D->hasDefaultArgument() && !D->defaultArgumentWasInherited())
    if (Expr *DefArg = D->getDefaultArgument())
      return Visit(MakeCXCursor(DefArg, StmtParent, TU, RegionOfInterest));
  
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
    if (Visit(MakeCXCursor(*P, TU, RegionOfInterest)))
      return true;
  }

  if (ND->isThisDeclarationADefinition() &&
      Visit(MakeCXCursor(ND->getBody(), StmtParent, TU, RegionOfInterest)))
    return true;

  return false;
}

template <typename DeclIt>
static void addRangedDeclsInContainer(DeclIt *DI_current, DeclIt DE_current,
                                      SourceManager &SM, SourceLocation EndLoc,
                                      SmallVectorImpl<Decl *> &Decls) {
  DeclIt next = *DI_current;
  while (++next != DE_current) {
    Decl *D_next = *next;
    if (!D_next)
      break;
    SourceLocation L = D_next->getLocStart();
    if (!L.isValid())
      break;
    if (SM.isBeforeInTranslationUnit(L, EndLoc)) {
      *DI_current = next;
      Decls.push_back(D_next);
      continue;
    }
    break;
  }
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
  if (!DI_current && !FileDI_current)
    return VisitDeclContext(D);

  // Scan the Decls that immediately come after the container
  // in the current DeclContext.  If any fall within the
  // container's lexical region, stash them into a vector
  // for later processing.
  SmallVector<Decl *, 24> DeclsInContainer;
  SourceLocation EndLoc = D->getSourceRange().getEnd();
  SourceManager &SM = AU->getSourceManager();
  if (EndLoc.isValid()) {
    if (DI_current) {
      addRangedDeclsInContainer(DI_current, DE_current, SM, EndLoc,
                                DeclsInContainer);
    } else {
      addRangedDeclsInContainer(FileDI_current, FileDE_current, SM, EndLoc,
                                DeclsInContainer);
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
  for (SmallVectorImpl<Decl*>::iterator I = DeclsInContainer.begin(),
         E = DeclsInContainer.end(); I != E; ++I) {
    CXCursor Cursor = MakeCXCursor(*I, TU, RegionOfInterest);
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
  if (!PID->isThisDeclarationADefinition())
    return Visit(MakeCursorObjCProtocolRef(PID, PID->getLocation(), TU));
  
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
    if (MD->isPropertyAccessor() && MD->getLexicalDeclContext() == CDecl)
      if (Visit(MakeCXCursor(MD, TU, RegionOfInterest)))
        return true;

  if (ObjCMethodDecl *MD = prevDecl->getSetterMethodDecl())
    if (MD->isPropertyAccessor() && MD->getLexicalDeclContext() == CDecl)
      if (Visit(MakeCXCursor(MD, TU, RegionOfInterest)))
        return true;

  return false;
}

bool CursorVisitor::VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {
  if (!D->isThisDeclarationADefinition()) {
    // Forward declaration is treated like a reference.
    return Visit(MakeCursorObjCClassRef(D, D->getLocation(), TU));
  }

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

bool CursorVisitor::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *PD) {
  if (ObjCIvarDecl *Ivar = PD->getPropertyIvarDecl())
    if (PD->isIvarNameSpecified())
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

  llvm_unreachable("Invalid DeclarationName::Kind!");
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
  SmallVector<NestedNameSpecifierLoc, 4> Qualifiers;
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
    if (Visit(MakeCXCursor(*P, TU, RegionOfInterest)))
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

  case TemplateName::SubstTemplateTemplateParm:
    return Visit(MakeCursorTemplateRef(
                         Name.getAsSubstTemplateTemplateParm()->getParameter(),
                                       Loc, TU));
      
  case TemplateName::SubstTemplateTemplateParmPack:
    return Visit(MakeCursorTemplateRef(
                  Name.getAsSubstTemplateTemplateParmPack()->getParameterPack(),
                                       Loc, TU));
  }

  llvm_unreachable("Invalid TemplateName::Kind!");
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
      return Visit(MakeCXCursor(E, StmtParent, TU, RegionOfInterest));
    return false;

  case TemplateArgument::NullPtr:
    if (Expr *E = TAL.getSourceNullPtrExpression())
      return Visit(MakeCXCursor(E, StmtParent, TU, RegionOfInterest));
    return false;

  case TemplateArgument::Expression:
    if (Expr *E = TAL.getSourceExpression())
      return Visit(MakeCXCursor(E, StmtParent, TU, RegionOfInterest));
    return false;
  
  case TemplateArgument::Template:
  case TemplateArgument::TemplateExpansion:
    if (VisitNestedNameSpecifierLoc(TAL.getTemplateQualifierLoc()))
      return true;
      
    return VisitTemplateName(TAL.getArgument().getAsTemplateOrTemplatePattern(), 
                             TAL.getTemplateNameLoc());
  }

  llvm_unreachable("Invalid TemplateArgument::Kind!");
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
  switch (TL.getTypePtr()->getKind()) {

  case BuiltinType::Void:
  case BuiltinType::NullPtr:
  case BuiltinType::Dependent:
#define BUILTIN_TYPE(Id, SingletonId)
#define SIGNED_TYPE(Id, SingletonId) case BuiltinType::Id:
#define UNSIGNED_TYPE(Id, SingletonId) case BuiltinType::Id:
#define FLOATING_TYPE(Id, SingletonId) case BuiltinType::Id:
#define PLACEHOLDER_TYPE(Id, SingletonId) case BuiltinType::Id:
#include "clang/AST/BuiltinTypes.def"
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
  return Visit(MakeCursorTypeRef(TL.getTypedefNameDecl(), TL.getNameLoc(), TU));
}

bool CursorVisitor::VisitUnresolvedUsingTypeLoc(UnresolvedUsingTypeLoc TL) {
  return Visit(MakeCursorTypeRef(TL.getDecl(), TL.getNameLoc(), TU));
}

bool CursorVisitor::VisitTagTypeLoc(TagTypeLoc TL) {
  if (TL.isDefinition())
    return Visit(MakeCXCursor(TL.getDecl(), TU, RegionOfInterest));

  return Visit(MakeCursorTypeRef(TL.getDecl(), TL.getNameLoc(), TU));
}

bool CursorVisitor::VisitTemplateTypeParmTypeLoc(TemplateTypeParmTypeLoc TL) {
  return Visit(MakeCursorTypeRef(TL.getDecl(), TL.getNameLoc(), TU));
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

bool CursorVisitor::VisitAttributedTypeLoc(AttributedTypeLoc TL) {
  return Visit(TL.getModifiedLoc());
}

bool CursorVisitor::VisitFunctionTypeLoc(FunctionTypeLoc TL, 
                                         bool SkipResultType) {
  if (!SkipResultType && Visit(TL.getResultLoc()))
    return true;

  for (unsigned I = 0, N = TL.getNumArgs(); I != N; ++I)
    if (Decl *D = TL.getArg(I))
      if (Visit(MakeCXCursor(D, TU, RegionOfInterest)))
        return true;

  return false;
}

bool CursorVisitor::VisitArrayTypeLoc(ArrayTypeLoc TL) {
  if (Visit(TL.getElementLoc()))
    return true;

  if (Expr *Size = TL.getSizeExpr())
    return Visit(MakeCXCursor(Size, StmtParent, TU, RegionOfInterest));

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

bool CursorVisitor::VisitUnaryTransformTypeLoc(UnaryTransformTypeLoc TL) {
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

bool CursorVisitor::VisitDecltypeTypeLoc(DecltypeTypeLoc TL) {
  if (Expr *E = TL.getUnderlyingExpr())
    return Visit(MakeCXCursor(E, StmtParent, TU));

  return false;
}

bool CursorVisitor::VisitInjectedClassNameTypeLoc(InjectedClassNameTypeLoc TL) {
  return Visit(MakeCursorTypeRef(TL.getDecl(), TL.getNameLoc(), TU));
}

bool CursorVisitor::VisitAtomicTypeLoc(AtomicTypeLoc TL) {
  return Visit(TL.getValueLoc());
}

#define DEFAULT_TYPELOC_IMPL(CLASS, PARENT) \
bool CursorVisitor::Visit##CLASS##TypeLoc(CLASS##TypeLoc TL) { \
  return Visit##PARENT##Loc(TL); \
}

DEFAULT_TYPELOC_IMPL(Complex, Type)
DEFAULT_TYPELOC_IMPL(ConstantArray, ArrayType)
DEFAULT_TYPELOC_IMPL(IncompleteArray, ArrayType)
DEFAULT_TYPELOC_IMPL(VariableArray, ArrayType)
DEFAULT_TYPELOC_IMPL(DependentSizedArray, ArrayType)
DEFAULT_TYPELOC_IMPL(DependentSizedExtVector, Type)
DEFAULT_TYPELOC_IMPL(Vector, Type)
DEFAULT_TYPELOC_IMPL(ExtVector, VectorType)
DEFAULT_TYPELOC_IMPL(FunctionProto, FunctionType)
DEFAULT_TYPELOC_IMPL(FunctionNoProto, FunctionType)
DEFAULT_TYPELOC_IMPL(Record, TagType)
DEFAULT_TYPELOC_IMPL(Enum, TagType)
DEFAULT_TYPELOC_IMPL(SubstTemplateTypeParm, Type)
DEFAULT_TYPELOC_IMPL(SubstTemplateTypeParmPack, Type)
DEFAULT_TYPELOC_IMPL(Auto, Type)

bool CursorVisitor::VisitCXXRecordDecl(CXXRecordDecl *D) {
  // Visit the nested-name-specifier, if present.
  if (NestedNameSpecifierLoc QualifierLoc = D->getQualifierLoc())
    if (VisitNestedNameSpecifierLoc(QualifierLoc))
      return true;

  if (D->isCompleteDefinition()) {
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
DEF_JOB(ExplicitTemplateArgsVisit, ASTTemplateArgumentListInfo, 
        ExplicitTemplateArgsVisitKind)
DEF_JOB(SizeOfPackExprParts, SizeOfPackExpr, SizeOfPackExprPartsKind)
DEF_JOB(LambdaExprParts, LambdaExpr, LambdaExprPartsKind)
DEF_JOB(PostChildrenVisit, void, PostChildrenVisitKind)
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
    case clang::Stmt::MSDependentExistsStmtClass:
      return cast<MSDependentExistsStmt>(S)->getNameInfo();
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
  void VisitMSDependentExistsStmt(MSDependentExistsStmt *S);
  void VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr *E);
  void VisitCXXNewExpr(CXXNewExpr *E);
  void VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E);
  void VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E);
  void VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *E);
  void VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *E);
  void VisitCXXTypeidExpr(CXXTypeidExpr *E);
  void VisitCXXUnresolvedConstructExpr(CXXUnresolvedConstructExpr *E);
  void VisitCXXUuidofExpr(CXXUuidofExpr *E);
  void VisitCXXCatchStmt(CXXCatchStmt *S);
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
  void VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *E);
  void VisitStmt(Stmt *S);
  void VisitSwitchStmt(SwitchStmt *S);
  void VisitWhileStmt(WhileStmt *W);
  void VisitUnaryTypeTraitExpr(UnaryTypeTraitExpr *E);
  void VisitBinaryTypeTraitExpr(BinaryTypeTraitExpr *E);
  void VisitTypeTraitExpr(TypeTraitExpr *E);
  void VisitArrayTypeTraitExpr(ArrayTypeTraitExpr *E);
  void VisitExpressionTraitExpr(ExpressionTraitExpr *E);
  void VisitUnresolvedMemberExpr(UnresolvedMemberExpr *U);
  void VisitVAArgExpr(VAArgExpr *E);
  void VisitSizeOfPackExpr(SizeOfPackExpr *E);
  void VisitPseudoObjectExpr(PseudoObjectExpr *E);
  void VisitOpaqueValueExpr(OpaqueValueExpr *E);
  void VisitLambdaExpr(LambdaExpr *E);
  
private:
  void AddDeclarationNameInfo(Stmt *S);
  void AddNestedNameSpecifierLoc(NestedNameSpecifierLoc Qualifier);
  void AddExplicitTemplateArgs(const ASTTemplateArgumentListInfo *A);
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
  AddExplicitTemplateArgs(const ASTTemplateArgumentListInfo *A) {
  if (A)
    WL.push_back(ExplicitTemplateArgsVisit(
                        const_cast<ASTTemplateArgumentListInfo*>(A), Parent));
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
VisitMSDependentExistsStmt(MSDependentExistsStmt *S) {
  AddStmt(S->getSubStmt());
  AddDeclarationNameInfo(S);
  if (NestedNameSpecifierLoc QualifierLoc = S->getQualifierLoc())
    AddNestedNameSpecifierLoc(QualifierLoc);
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
  // Enqueue the initializer , if any.
  AddStmt(E->getInitializer());
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

void EnqueueVisitor::VisitCXXCatchStmt(CXXCatchStmt *S) {
  EnqueueChildren(S);
  AddDecl(S->getExceptionDecl());
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
      AddMemberRef(Node.getField(), Node.getSourceRange().getEnd());
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
void EnqueueVisitor::VisitUnaryExprOrTypeTraitExpr(
                                              UnaryExprOrTypeTraitExpr *E) {
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

void EnqueueVisitor::VisitTypeTraitExpr(TypeTraitExpr *E) {
  for (unsigned I = E->getNumArgs(); I > 0; --I)
    AddTypeLoc(E->getArg(I-1));
}

void EnqueueVisitor::VisitArrayTypeTraitExpr(ArrayTypeTraitExpr *E) {
  AddTypeLoc(E->getQueriedTypeSourceInfo());
}

void EnqueueVisitor::VisitExpressionTraitExpr(ExpressionTraitExpr *E) {
  EnqueueChildren(E);
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
void EnqueueVisitor::VisitOpaqueValueExpr(OpaqueValueExpr *E) {
  // If the opaque value has a source expression, just transparently
  // visit that.  This is useful for (e.g.) pseudo-object expressions.
  if (Expr *SourceExpr = E->getSourceExpr())
    return Visit(SourceExpr);
}
void EnqueueVisitor::VisitLambdaExpr(LambdaExpr *E) {
  AddStmt(E->getBody());
  WL.push_back(LambdaExprParts(E, Parent));
}
void EnqueueVisitor::VisitPseudoObjectExpr(PseudoObjectExpr *E) {
  // Treat the expression like its syntactic form.
  Visit(E->getSyntacticForm());
}

void CursorVisitor::EnqueueWorkList(VisitorWorkList &WL, Stmt *S) {
  EnqueueVisitor(WL, MakeCXCursor(S, StmtParent, TU,RegionOfInterest)).Visit(S);
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
        if (Visit(MakeCXCursor(D, TU, RegionOfInterest,
                               cast<DeclVisit>(&LI)->isFirst())))
            return true;

        continue;
      }
      case VisitorJob::ExplicitTemplateArgsVisitKind: {
        const ASTTemplateArgumentListInfo *ArgList =
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
        CXCursor Cursor = MakeCXCursor(S, StmtParent, TU, RegionOfInterest);
        if (!IsInRegionOfInterest(Cursor))
          continue;
        switch (Visitor(Cursor, Parent, ClientData)) {
          case CXChildVisit_Break: return true;
          case CXChildVisit_Continue: break;
          case CXChildVisit_Recurse:
            if (PostChildrenVisitor)
              WL.push_back(PostChildrenVisit(0, Cursor));
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
        
      case VisitorJob::LambdaExprPartsKind: {
        // Visit captures.
        LambdaExpr *E = cast<LambdaExprParts>(&LI)->get();
        for (LambdaExpr::capture_iterator C = E->explicit_capture_begin(),
                                       CEnd = E->explicit_capture_end();
             C != CEnd; ++C) {
          if (C->capturesThis())
            continue;
          
          if (Visit(MakeCursorVariableRef(C->getCapturedVar(),
                                          C->getLocation(),
                                          TU)))
            return true;
        }
        
        // Visit parameters and return type, if present.
        if (E->hasExplicitParameters() || E->hasExplicitResultType()) {
          TypeLoc TL = E->getCallOperator()->getTypeSourceInfo()->getTypeLoc();
          if (E->hasExplicitParameters() && E->hasExplicitResultType()) {
            // Visit the whole type.
            if (Visit(TL))
              return true;
          } else if (isa<FunctionProtoTypeLoc>(TL)) {
            FunctionProtoTypeLoc Proto = cast<FunctionProtoTypeLoc>(TL);
            if (E->hasExplicitParameters()) {
              // Visit parameters.
              for (unsigned I = 0, N = Proto.getNumArgs(); I != N; ++I)
                if (Visit(MakeCXCursor(Proto.getArg(I), TU)))
                  return true;
            } else {
              // Visit result type.
              if (Visit(Proto.getResultLoc()))
                return true;
            }
          }
        }
        break;
      }

      case VisitorJob::PostChildrenVisitKind:
        if (PostChildrenVisitor(Parent, ClientData))
          return true;
        break;
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

namespace {
typedef llvm::SmallVector<SourceRange, 4> RefNamePieces;
RefNamePieces buildPieces(unsigned NameFlags, bool IsMemberRefExpr, 
                          const DeclarationNameInfo &NI, 
                          const SourceRange &QLoc, 
                          const ASTTemplateArgumentListInfo *TemplateArgs = 0){
  const bool WantQualifier = NameFlags & CXNameRange_WantQualifier;
  const bool WantTemplateArgs = NameFlags & CXNameRange_WantTemplateArgs;
  const bool WantSinglePiece = NameFlags & CXNameRange_WantSinglePiece;
  
  const DeclarationName::NameKind Kind = NI.getName().getNameKind();
  
  RefNamePieces Pieces;

  if (WantQualifier && QLoc.isValid())
    Pieces.push_back(QLoc);
  
  if (Kind != DeclarationName::CXXOperatorName || IsMemberRefExpr)
    Pieces.push_back(NI.getLoc());
  
  if (WantTemplateArgs && TemplateArgs)
    Pieces.push_back(SourceRange(TemplateArgs->LAngleLoc,
                                 TemplateArgs->RAngleLoc));
  
  if (Kind == DeclarationName::CXXOperatorName) {
    Pieces.push_back(SourceLocation::getFromRawEncoding(
                       NI.getInfo().CXXOperatorName.BeginOpNameLoc));
    Pieces.push_back(SourceLocation::getFromRawEncoding(
                       NI.getInfo().CXXOperatorName.EndOpNameLoc));
  }
  
  if (WantSinglePiece) {
    SourceRange R(Pieces.front().getBegin(), Pieces.back().getEnd());
    Pieces.clear();
    Pieces.push_back(R);
  }  

  return Pieces;  
}
}

//===----------------------------------------------------------------------===//
// Misc. API hooks.
//===----------------------------------------------------------------------===//               

static llvm::sys::Mutex EnableMultithreadingMutex;
static bool EnabledMultithreading;

static void fatal_error_handler(void *user_data, const std::string& reason) {
  // Write the result out to stderr avoiding errs() because raw_ostreams can
  // call report_fatal_error.
  fprintf(stderr, "LIBCLANG FATAL ERROR: %s\n", reason.c_str());
  ::abort();
}

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
      llvm::install_fatal_error_handler(fatal_error_handler, 0);
      llvm::llvm_start_multithreaded();
      EnabledMultithreading = true;
    }
  }

  CIndexer *CIdxr = new CIndexer();
  if (excludeDeclarationsFromPCH)
    CIdxr->setOnlyLocalDecls();
  if (displayDiagnostics)
    CIdxr->setDisplayDiagnostics();

  if (getenv("LIBCLANG_BGPRIO_INDEX"))
    CIdxr->setCXGlobalOptFlags(CIdxr->getCXGlobalOptFlags() |
                               CXGlobalOpt_ThreadBackgroundPriorityForIndexing);
  if (getenv("LIBCLANG_BGPRIO_EDIT"))
    CIdxr->setCXGlobalOptFlags(CIdxr->getCXGlobalOptFlags() |
                               CXGlobalOpt_ThreadBackgroundPriorityForEditing);

  return CIdxr;
}

void clang_disposeIndex(CXIndex CIdx) {
  if (CIdx)
    delete static_cast<CIndexer *>(CIdx);
}

void clang_CXIndex_setGlobalOptions(CXIndex CIdx, unsigned options) {
  if (CIdx)
    static_cast<CIndexer *>(CIdx)->setCXGlobalOptFlags(options);
}

unsigned clang_CXIndex_getGlobalOptions(CXIndex CIdx) {
  if (CIdx)
    return static_cast<CIndexer *>(CIdx)->getCXGlobalOptFlags();
  return 0;
}

void clang_toggleCrashRecovery(unsigned isEnabled) {
  if (isEnabled)
    llvm::CrashRecoveryContext::Enable();
  else
    llvm::CrashRecoveryContext::Disable();
}
  
CXTranslationUnit clang_createTranslationUnit(CXIndex CIdx,
                                              const char *ast_filename) {
  if (!CIdx)
    return 0;

  CIndexer *CXXIdx = static_cast<CIndexer *>(CIdx);
  FileSystemOptions FileSystemOpts;
  FileSystemOpts.WorkingDir = CXXIdx->getWorkingDirectory();

  IntrusiveRefCntPtr<DiagnosticsEngine> Diags;
  ASTUnit *TU = ASTUnit::LoadFromASTFile(ast_filename, Diags, FileSystemOpts,
                                  CXXIdx->getOnlyLocalDecls(),
                                  0, 0,
                                  /*CaptureDiagnostics=*/true,
                                  /*AllowPCHWithCompilerErrors=*/true,
                                  /*UserFilesAreVolatile=*/true);
  return MakeCXTranslationUnit(CXXIdx, TU);
}

unsigned clang_defaultEditingTranslationUnitOptions() {
  return CXTranslationUnit_PrecompiledPreamble | 
         CXTranslationUnit_CacheCompletionResults;
}
  
CXTranslationUnit
clang_createTranslationUnitFromSourceFile(CXIndex CIdx,
                                          const char *source_filename,
                                          int num_command_line_args,
                                          const char * const *command_line_args,
                                          unsigned num_unsaved_files,
                                          struct CXUnsavedFile *unsaved_files) {
  unsigned Options = CXTranslationUnit_DetailedPreprocessingRecord;
  return clang_parseTranslationUnit(CIdx, source_filename,
                                    command_line_args, num_command_line_args,
                                    unsaved_files, num_unsaved_files,
                                    Options);
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

  if (CXXIdx->isOptEnabled(CXGlobalOpt_ThreadBackgroundPriorityForIndexing))
    setThreadBackgroundPriority();

  bool PrecompilePreamble = options & CXTranslationUnit_PrecompiledPreamble;
  // FIXME: Add a flag for modules.
  TranslationUnitKind TUKind
    = (options & CXTranslationUnit_Incomplete)? TU_Prefix : TU_Complete;
  bool CacheCodeCompetionResults
    = options & CXTranslationUnit_CacheCompletionResults;
  bool IncludeBriefCommentsInCodeCompletion
    = options & CXTranslationUnit_IncludeBriefCommentsInCodeCompletion;
  bool SkipFunctionBodies = options & CXTranslationUnit_SkipFunctionBodies;
  bool ForSerialization = options & CXTranslationUnit_ForSerialization;

  // Configure the diagnostics.
  IntrusiveRefCntPtr<DiagnosticsEngine>
    Diags(CompilerInstance::createDiagnostics(new DiagnosticOptions,
                                              num_command_line_args,
                                              command_line_args));

  // Recover resources if we crash before exiting this function.
  llvm::CrashRecoveryContextCleanupRegistrar<DiagnosticsEngine,
    llvm::CrashRecoveryContextReleaseRefCleanup<DiagnosticsEngine> >
    DiagCleanup(Diags.getPtr());

  OwningPtr<std::vector<ASTUnit::RemappedFile> >
    RemappedFiles(new std::vector<ASTUnit::RemappedFile>());

  // Recover resources if we crash before exiting this function.
  llvm::CrashRecoveryContextCleanupRegistrar<
    std::vector<ASTUnit::RemappedFile> > RemappedCleanup(RemappedFiles.get());

  for (unsigned I = 0; I != num_unsaved_files; ++I) {
    StringRef Data(unsaved_files[I].Contents, unsaved_files[I].Length);
    const llvm::MemoryBuffer *Buffer
      = llvm::MemoryBuffer::getMemBufferCopy(Data, unsaved_files[I].Filename);
    RemappedFiles->push_back(std::make_pair(unsaved_files[I].Filename,
                                            Buffer));
  }

  OwningPtr<std::vector<const char *> >
    Args(new std::vector<const char*>());

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<std::vector<const char*> >
    ArgsCleanup(Args.get());

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
    Args->push_back("-fno-spell-checking");
  
  Args->insert(Args->end(), command_line_args,
               command_line_args + num_command_line_args);

  // The 'source_filename' argument is optional.  If the caller does not
  // specify it then it is assumed that the source file is specified
  // in the actual argument list.
  // Put the source file after command_line_args otherwise if '-x' flag is
  // present it will be unused.
  if (source_filename)
    Args->push_back(source_filename);

  // Do we need the detailed preprocessing record?
  if (options & CXTranslationUnit_DetailedPreprocessingRecord) {
    Args->push_back("-Xclang");
    Args->push_back("-detailed-preprocessing-record");
  }
  
  unsigned NumErrors = Diags->getClient()->getNumErrors();
  OwningPtr<ASTUnit> ErrUnit;
  OwningPtr<ASTUnit> Unit(
    ASTUnit::LoadFromCommandLine(Args->size() ? &(*Args)[0] : 0 
                                 /* vector::data() not portable */,
                                 Args->size() ? (&(*Args)[0] + Args->size()) :0,
                                 Diags,
                                 CXXIdx->getClangResourcesPath(),
                                 CXXIdx->getOnlyLocalDecls(),
                                 /*CaptureDiagnostics=*/true,
                                 RemappedFiles->size() ? &(*RemappedFiles)[0]:0,
                                 RemappedFiles->size(),
                                 /*RemappedFilesKeepOriginalName=*/true,
                                 PrecompilePreamble,
                                 TUKind,
                                 CacheCodeCompetionResults,
                                 IncludeBriefCommentsInCodeCompletion,
                                 /*AllowPCHWithCompilerErrors=*/true,
                                 SkipFunctionBodies,
                                 /*UserFilesAreVolatile=*/true,
                                 ForSerialization,
                                 &ErrUnit));

  if (NumErrors != Diags->getClient()->getNumErrors()) {
    // Make sure to check that 'Unit' is non-NULL.
    if (CXXIdx->getDisplayDiagnostics())
      printDiagsToStderr(Unit ? Unit.get() : ErrUnit.get());
  }

  PTUI->result = MakeCXTranslationUnit(CXXIdx, Unit.take());
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
  } else if (getenv("LIBCLANG_RESOURCE_USAGE")) {
    PrintLibclangResourceUsage(PTUI.result);
  }
  
  return PTUI.result;
}

unsigned clang_defaultSaveOptions(CXTranslationUnit TU) {
  return CXSaveTranslationUnit_None;
}  

namespace {

struct SaveTranslationUnitInfo {
  CXTranslationUnit TU;
  const char *FileName;
  unsigned options;
  CXSaveError result;
};

}

static void clang_saveTranslationUnit_Impl(void *UserData) {
  SaveTranslationUnitInfo *STUI =
    static_cast<SaveTranslationUnitInfo*>(UserData);

  CIndexer *CXXIdx = (CIndexer*)STUI->TU->CIdx;
  if (CXXIdx->isOptEnabled(CXGlobalOpt_ThreadBackgroundPriorityForIndexing))
    setThreadBackgroundPriority();

  bool hadError = static_cast<ASTUnit *>(STUI->TU->TUData)->Save(STUI->FileName);
  STUI->result = hadError ? CXSaveError_Unknown : CXSaveError_None;
}

int clang_saveTranslationUnit(CXTranslationUnit TU, const char *FileName,
                              unsigned options) {
  if (!TU)
    return CXSaveError_InvalidTU;

  ASTUnit *CXXUnit = static_cast<ASTUnit *>(TU->TUData);
  ASTUnit::ConcurrencyCheck Check(*CXXUnit);
  if (!CXXUnit->hasSema())
    return CXSaveError_InvalidTU;

  SaveTranslationUnitInfo STUI = { TU, FileName, options, CXSaveError_None };

  if (!CXXUnit->getDiagnostics().hasUnrecoverableErrorOccurred() ||
      getenv("LIBCLANG_NOTHREADS")) {
    clang_saveTranslationUnit_Impl(&STUI);

    if (getenv("LIBCLANG_RESOURCE_USAGE"))
      PrintLibclangResourceUsage(TU);

    return STUI.result;
  }

  // We have an AST that has invalid nodes due to compiler errors.
  // Use a crash recovery thread for protection.

  llvm::CrashRecoveryContext CRC;

  if (!RunSafely(CRC, clang_saveTranslationUnit_Impl, &STUI)) {
    fprintf(stderr, "libclang: crash detected during AST saving: {\n");
    fprintf(stderr, "  'filename' : '%s'\n", FileName);
    fprintf(stderr, "  'options' : %d,\n", options);
    fprintf(stderr, "}\n");

    return CXSaveError_Unknown;

  } else if (getenv("LIBCLANG_RESOURCE_USAGE")) {
    PrintLibclangResourceUsage(TU);
  }

  return STUI.result;
}

void clang_disposeTranslationUnit(CXTranslationUnit CTUnit) {
  if (CTUnit) {
    // If the translation unit has been marked as unsafe to free, just discard
    // it.
    if (static_cast<ASTUnit *>(CTUnit->TUData)->isUnsafeToFree())
      return;

    delete static_cast<ASTUnit *>(CTUnit->TUData);
    disposeCXStringPool(CTUnit->StringPool);
    delete static_cast<CXDiagnosticSetImpl *>(CTUnit->Diagnostics);
    disposeOverridenCXCursorsPool(CTUnit->OverridenCursorsPool);
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

  // Reset the associated diagnostics.
  delete static_cast<CXDiagnosticSetImpl*>(TU->Diagnostics);
  TU->Diagnostics = 0;

  unsigned num_unsaved_files = RTUI->num_unsaved_files;
  struct CXUnsavedFile *unsaved_files = RTUI->unsaved_files;
  unsigned options = RTUI->options;
  (void) options;
  RTUI->result = 1;

  if (!TU)
    return;

  CIndexer *CXXIdx = (CIndexer*)TU->CIdx;
  if (CXXIdx->isOptEnabled(CXGlobalOpt_ThreadBackgroundPriorityForEditing))
    setThreadBackgroundPriority();

  ASTUnit *CXXUnit = static_cast<ASTUnit *>(TU->TUData);
  ASTUnit::ConcurrencyCheck Check(*CXXUnit);
  
  OwningPtr<std::vector<ASTUnit::RemappedFile> >
    RemappedFiles(new std::vector<ASTUnit::RemappedFile>());
  
  // Recover resources if we crash before exiting this function.
  llvm::CrashRecoveryContextCleanupRegistrar<
    std::vector<ASTUnit::RemappedFile> > RemappedCleanup(RemappedFiles.get());
  
  for (unsigned I = 0; I != num_unsaved_files; ++I) {
    StringRef Data(unsaved_files[I].Contents, unsaved_files[I].Length);
    const llvm::MemoryBuffer *Buffer
      = llvm::MemoryBuffer::getMemBufferCopy(Data, unsaved_files[I].Filename);
    RemappedFiles->push_back(std::make_pair(unsaved_files[I].Filename,
                                            Buffer));
  }
  
  if (!CXXUnit->Reparse(RemappedFiles->size() ? &(*RemappedFiles)[0] : 0,
                        RemappedFiles->size()))
    RTUI->result = 0;
}

int clang_reparseTranslationUnit(CXTranslationUnit TU,
                                 unsigned num_unsaved_files,
                                 struct CXUnsavedFile *unsaved_files,
                                 unsigned options) {
  ReparseTranslationUnitInfo RTUI = { TU, num_unsaved_files, unsaved_files,
                                      options, 0 };

  if (getenv("LIBCLANG_NOTHREADS")) {
    clang_reparseTranslationUnit_Impl(&RTUI);
    return RTUI.result;
  }

  llvm::CrashRecoveryContext CRC;

  if (!RunSafely(CRC, clang_reparseTranslationUnit_Impl, &RTUI)) {
    fprintf(stderr, "libclang: crash detected during reparsing\n");
    static_cast<ASTUnit *>(TU->TUData)->setUnsafeToFree(true);
    return 1;
  } else if (getenv("LIBCLANG_RESOURCE_USAGE"))
    PrintLibclangResourceUsage(TU);

  return RTUI.result;
}


CXString clang_getTranslationUnitSpelling(CXTranslationUnit CTUnit) {
  if (!CTUnit)
    return createCXString("");

  ASTUnit *CXXUnit = static_cast<ASTUnit *>(CTUnit->TUData);
  return createCXString(CXXUnit->getOriginalSourceFileName(), true);
}

CXCursor clang_getTranslationUnitCursor(CXTranslationUnit TU) {
  ASTUnit *CXXUnit = static_cast<ASTUnit*>(TU->TUData);
  return MakeCXCursor(CXXUnit->getASTContext().getTranslationUnitDecl(), TU);
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

unsigned clang_isFileMultipleIncludeGuarded(CXTranslationUnit tu, CXFile file) {
  if (!tu || !file)
    return 0;

  ASTUnit *CXXUnit = static_cast<ASTUnit *>(tu->TUData);
  FileEntry *FEnt = static_cast<FileEntry *>(file);
  return CXXUnit->getPreprocessor().getHeaderSearchInfo()
                                          .isFileMultipleIncludeGuarded(FEnt);
}

} // end: extern "C"

//===----------------------------------------------------------------------===//
// CXCursor Operations.
//===----------------------------------------------------------------------===//

static Decl *getDeclFromExpr(Stmt *E) {
  if (ImplicitCastExpr *CE = dyn_cast<ImplicitCastExpr>(E))
    return getDeclFromExpr(CE->getSubExpr());

  if (DeclRefExpr *RefExpr = dyn_cast<DeclRefExpr>(E))
    return RefExpr->getDecl();
  if (MemberExpr *ME = dyn_cast<MemberExpr>(E))
    return ME->getMemberDecl();
  if (ObjCIvarRefExpr *RE = dyn_cast<ObjCIvarRefExpr>(E))
    return RE->getDecl();
  if (ObjCPropertyRefExpr *PRE = dyn_cast<ObjCPropertyRefExpr>(E)) {
    if (PRE->isExplicitProperty())
      return PRE->getExplicitProperty();
    // It could be messaging both getter and setter as in:
    // ++myobj.myprop;
    // in which case prefer to associate the setter since it is less obvious
    // from inspecting the source that the setter is going to get called.
    if (PRE->isMessagingSetter())
      return PRE->getImplicitPropertySetter();
    return PRE->getImplicitPropertyGetter();
  }
  if (PseudoObjectExpr *POE = dyn_cast<PseudoObjectExpr>(E))
    return getDeclFromExpr(POE->getSyntacticForm());
  if (OpaqueValueExpr *OVE = dyn_cast<OpaqueValueExpr>(E))
    if (Expr *Src = OVE->getSourceExpr())
      return getDeclFromExpr(Src);
      
  if (CallExpr *CE = dyn_cast<CallExpr>(E))
    return getDeclFromExpr(CE->getCallee());
  if (CXXConstructExpr *CE = dyn_cast<CXXConstructExpr>(E))
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
  if (ImplicitCastExpr *CE = dyn_cast<ImplicitCastExpr>(E))
    return getLocationFromExpr(CE->getSubExpr());

  if (ObjCMessageExpr *Msg = dyn_cast<ObjCMessageExpr>(E))
    return /*FIXME:*/Msg->getLeftLoc();
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E))
    return DRE->getLocation();
  if (MemberExpr *Member = dyn_cast<MemberExpr>(E))
    return Member->getMemberLoc();
  if (ObjCIvarRefExpr *Ivar = dyn_cast<ObjCIvarRefExpr>(E))
    return Ivar->getLocation();
  if (SizeOfPackExpr *SizeOfPack = dyn_cast<SizeOfPackExpr>(E))
    return SizeOfPack->getPackLoc();
  if (ObjCPropertyRefExpr *PropRef = dyn_cast<ObjCPropertyRefExpr>(E))
    return PropRef->getLocation();
  
  return E->getLocStart();
}

extern "C" {

unsigned clang_visitChildren(CXCursor parent,
                             CXCursorVisitor visitor,
                             CXClientData client_data) {
  CursorVisitor CursorVis(getCursorTU(parent), visitor, client_data,
                          /*VisitPreprocessorLast=*/false);
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
  if (!D)
    return createCXString("");

  NamedDecl *ND = dyn_cast<NamedDecl>(D);
  if (!ND) {
    if (ObjCPropertyImplDecl *PropImpl =dyn_cast<ObjCPropertyImplDecl>(D))
      if (ObjCPropertyDecl *Property = PropImpl->getPropertyDecl())
        return createCXString(Property->getIdentifier()->getName());
    
    if (ImportDecl *ImportD = dyn_cast<ImportDecl>(D))
      if (Module *Mod = ImportD->getImportedModule())
        return createCXString(Mod->getFullModuleName());

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
  
  SmallString<1024> S;
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
        
    case CXCursor_VariableRef: {
      VarDecl *Var = getCursorVariableRef(C).first;
      assert(Var && "Missing variable decl");
      
      return createCXString(Var->getNameAsString());
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
  
  if (C.kind == CXCursor_MacroExpansion)
    return createCXString(getCursorMacroExpansion(C)->getName()
                                                           ->getNameStart());

  if (C.kind == CXCursor_MacroDefinition)
    return createCXString(getCursorMacroDefinition(C)->getName()
                                                           ->getNameStart());

  if (C.kind == CXCursor_InclusionDirective)
    return createCXString(getCursorInclusionDirective(C)->getFileName());
      
  if (clang_isDeclaration(C.kind))
    return getDeclSpelling(getCursorDecl(C));

  if (C.kind == CXCursor_AnnotateAttr) {
    AnnotateAttr *AA = cast<AnnotateAttr>(cxcursor::getCursorAttr(C));
    return createCXString(AA->getAnnotation());
  }

  if (C.kind == CXCursor_AsmLabelAttr) {
    AsmLabelAttr *AA = cast<AsmLabelAttr>(cxcursor::getCursorAttr(C));
    return createCXString(AA->getLabel());
  }

  return createCXString("");
}

CXSourceRange clang_Cursor_getSpellingNameRange(CXCursor C,
                                                unsigned pieceIndex,
                                                unsigned options) {
  if (clang_Cursor_isNull(C))
    return clang_getNullRange();

  ASTContext &Ctx = getCursorContext(C);

  if (clang_isStatement(C.kind)) {
    Stmt *S = getCursorStmt(C);
    if (LabelStmt *Label = dyn_cast_or_null<LabelStmt>(S)) {
      if (pieceIndex > 0)
        return clang_getNullRange();
      return cxloc::translateSourceRange(Ctx, Label->getIdentLoc());
    }

    return clang_getNullRange();
  }

  if (C.kind == CXCursor_ObjCMessageExpr) {
    if (ObjCMessageExpr *
          ME = dyn_cast_or_null<ObjCMessageExpr>(getCursorExpr(C))) {
      if (pieceIndex >= ME->getNumSelectorLocs())
        return clang_getNullRange();
      return cxloc::translateSourceRange(Ctx, ME->getSelectorLoc(pieceIndex));
    }
  }

  if (C.kind == CXCursor_ObjCInstanceMethodDecl ||
      C.kind == CXCursor_ObjCClassMethodDecl) {
    if (ObjCMethodDecl *
          MD = dyn_cast_or_null<ObjCMethodDecl>(getCursorDecl(C))) {
      if (pieceIndex >= MD->getNumSelectorLocs())
        return clang_getNullRange();
      return cxloc::translateSourceRange(Ctx, MD->getSelectorLoc(pieceIndex));
    }
  }

  if (C.kind == CXCursor_ObjCCategoryDecl ||
      C.kind == CXCursor_ObjCCategoryImplDecl) {
    if (pieceIndex > 0)
      return clang_getNullRange();
    if (ObjCCategoryDecl *
          CD = dyn_cast_or_null<ObjCCategoryDecl>(getCursorDecl(C)))
      return cxloc::translateSourceRange(Ctx, CD->getCategoryNameLoc());
    if (ObjCCategoryImplDecl *
          CID = dyn_cast_or_null<ObjCCategoryImplDecl>(getCursorDecl(C)))
      return cxloc::translateSourceRange(Ctx, CID->getCategoryNameLoc());
  }

  if (C.kind == CXCursor_ModuleImportDecl) {
    if (pieceIndex > 0)
      return clang_getNullRange();
    if (ImportDecl *ImportD = dyn_cast_or_null<ImportDecl>(getCursorDecl(C))) {
      ArrayRef<SourceLocation> Locs = ImportD->getIdentifierLocs();
      if (!Locs.empty())
        return cxloc::translateSourceRange(Ctx,
                                         SourceRange(Locs.front(), Locs.back()));
    }
    return clang_getNullRange();
  }

  // FIXME: A CXCursor_InclusionDirective should give the location of the
  // filename, but we don't keep track of this.

  // FIXME: A CXCursor_AnnotateAttr should give the location of the annotation
  // but we don't keep track of this.

  // FIXME: A CXCursor_AsmLabelAttr should give the location of the label
  // but we don't keep track of this.

  // Default handling, give the location of the cursor.

  if (pieceIndex > 0)
    return clang_getNullRange();

  CXSourceLocation CXLoc = clang_getCursorLocation(C);
  SourceLocation Loc = cxloc::translateSourceLocation(CXLoc);
  return cxloc::translateSourceRange(Ctx, Loc);
}

CXString clang_getCursorDisplayName(CXCursor C) {
  if (!clang_isDeclaration(C.kind))
    return clang_getCursorSpelling(C);
  
  Decl *D = getCursorDecl(C);
  if (!D)
    return createCXString("");

  PrintingPolicy Policy = getCursorContext(C).getPrintingPolicy();
  if (FunctionTemplateDecl *FunTmpl = dyn_cast<FunctionTemplateDecl>(D))
    D = FunTmpl->getTemplatedDecl();
  
  if (FunctionDecl *Function = dyn_cast<FunctionDecl>(D)) {
    SmallString<64> Str;
    llvm::raw_svector_ostream OS(Str);
    OS << *Function;
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
    SmallString<64> Str;
    llvm::raw_svector_ostream OS(Str);
    OS << *ClassTemplate;
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
    
    SmallString<64> Str;
    llvm::raw_svector_ostream OS(Str);
    OS << *ClassSpec;
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
  case CXCursor_VariableRef:
    return createCXString("VariableRef");
  case CXCursor_IntegerLiteral:
      return createCXString("IntegerLiteral");
  case CXCursor_FloatingLiteral:
      return createCXString("FloatingLiteral");
  case CXCursor_ImaginaryLiteral:
      return createCXString("ImaginaryLiteral");
  case CXCursor_StringLiteral:
      return createCXString("StringLiteral");
  case CXCursor_CharacterLiteral:
      return createCXString("CharacterLiteral");
  case CXCursor_ParenExpr:
      return createCXString("ParenExpr");
  case CXCursor_UnaryOperator:
      return createCXString("UnaryOperator");
  case CXCursor_ArraySubscriptExpr:
      return createCXString("ArraySubscriptExpr");
  case CXCursor_BinaryOperator:
      return createCXString("BinaryOperator");
  case CXCursor_CompoundAssignOperator:
      return createCXString("CompoundAssignOperator");
  case CXCursor_ConditionalOperator:
      return createCXString("ConditionalOperator");
  case CXCursor_CStyleCastExpr:
      return createCXString("CStyleCastExpr");
  case CXCursor_CompoundLiteralExpr:
      return createCXString("CompoundLiteralExpr");
  case CXCursor_InitListExpr:
      return createCXString("InitListExpr");
  case CXCursor_AddrLabelExpr:
      return createCXString("AddrLabelExpr");
  case CXCursor_StmtExpr:
      return createCXString("StmtExpr");
  case CXCursor_GenericSelectionExpr:
      return createCXString("GenericSelectionExpr");
  case CXCursor_GNUNullExpr:
      return createCXString("GNUNullExpr");
  case CXCursor_CXXStaticCastExpr:
      return createCXString("CXXStaticCastExpr");
  case CXCursor_CXXDynamicCastExpr:
      return createCXString("CXXDynamicCastExpr");
  case CXCursor_CXXReinterpretCastExpr:
      return createCXString("CXXReinterpretCastExpr");
  case CXCursor_CXXConstCastExpr:
      return createCXString("CXXConstCastExpr");
  case CXCursor_CXXFunctionalCastExpr:
      return createCXString("CXXFunctionalCastExpr");
  case CXCursor_CXXTypeidExpr:
      return createCXString("CXXTypeidExpr");
  case CXCursor_CXXBoolLiteralExpr:
      return createCXString("CXXBoolLiteralExpr");
  case CXCursor_CXXNullPtrLiteralExpr:
      return createCXString("CXXNullPtrLiteralExpr");
  case CXCursor_CXXThisExpr:
      return createCXString("CXXThisExpr");
  case CXCursor_CXXThrowExpr:
      return createCXString("CXXThrowExpr");
  case CXCursor_CXXNewExpr:
      return createCXString("CXXNewExpr");
  case CXCursor_CXXDeleteExpr:
      return createCXString("CXXDeleteExpr");
  case CXCursor_UnaryExpr:
      return createCXString("UnaryExpr");
  case CXCursor_ObjCStringLiteral:
      return createCXString("ObjCStringLiteral");
  case CXCursor_ObjCBoolLiteralExpr:
      return createCXString("ObjCBoolLiteralExpr");
  case CXCursor_ObjCEncodeExpr:
      return createCXString("ObjCEncodeExpr");
  case CXCursor_ObjCSelectorExpr:
      return createCXString("ObjCSelectorExpr");
  case CXCursor_ObjCProtocolExpr:
      return createCXString("ObjCProtocolExpr");
  case CXCursor_ObjCBridgedCastExpr:
      return createCXString("ObjCBridgedCastExpr");
  case CXCursor_BlockExpr:
      return createCXString("BlockExpr");
  case CXCursor_PackExpansionExpr:
      return createCXString("PackExpansionExpr");
  case CXCursor_SizeOfPackExpr:
      return createCXString("SizeOfPackExpr");
  case CXCursor_LambdaExpr:
    return createCXString("LambdaExpr");
  case CXCursor_UnexposedExpr:
      return createCXString("UnexposedExpr");
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
  case CXCursor_DeclStmt:
      return createCXString("DeclStmt");
  case CXCursor_LabelStmt:
      return createCXString("LabelStmt");
  case CXCursor_CompoundStmt:
      return createCXString("CompoundStmt");
  case CXCursor_CaseStmt:
      return createCXString("CaseStmt");
  case CXCursor_DefaultStmt:
      return createCXString("DefaultStmt");
  case CXCursor_IfStmt:
      return createCXString("IfStmt");
  case CXCursor_SwitchStmt:
      return createCXString("SwitchStmt");
  case CXCursor_WhileStmt:
      return createCXString("WhileStmt");
  case CXCursor_DoStmt:
      return createCXString("DoStmt");
  case CXCursor_ForStmt:
      return createCXString("ForStmt");
  case CXCursor_GotoStmt:
      return createCXString("GotoStmt");
  case CXCursor_IndirectGotoStmt:
      return createCXString("IndirectGotoStmt");
  case CXCursor_ContinueStmt:
      return createCXString("ContinueStmt");
  case CXCursor_BreakStmt:
      return createCXString("BreakStmt");
  case CXCursor_ReturnStmt:
      return createCXString("ReturnStmt");
  case CXCursor_GCCAsmStmt:
      return createCXString("GCCAsmStmt");
  case CXCursor_MSAsmStmt:
      return createCXString("MSAsmStmt");
  case CXCursor_ObjCAtTryStmt:
      return createCXString("ObjCAtTryStmt");
  case CXCursor_ObjCAtCatchStmt:
      return createCXString("ObjCAtCatchStmt");
  case CXCursor_ObjCAtFinallyStmt:
      return createCXString("ObjCAtFinallyStmt");
  case CXCursor_ObjCAtThrowStmt:
      return createCXString("ObjCAtThrowStmt");
  case CXCursor_ObjCAtSynchronizedStmt:
      return createCXString("ObjCAtSynchronizedStmt");
  case CXCursor_ObjCAutoreleasePoolStmt:
      return createCXString("ObjCAutoreleasePoolStmt");
  case CXCursor_ObjCForCollectionStmt:
      return createCXString("ObjCForCollectionStmt");
  case CXCursor_CXXCatchStmt:
      return createCXString("CXXCatchStmt");
  case CXCursor_CXXTryStmt:
      return createCXString("CXXTryStmt");
  case CXCursor_CXXForRangeStmt:
      return createCXString("CXXForRangeStmt");
  case CXCursor_SEHTryStmt:
      return createCXString("SEHTryStmt");
  case CXCursor_SEHExceptStmt:
      return createCXString("SEHExceptStmt");
  case CXCursor_SEHFinallyStmt:
      return createCXString("SEHFinallyStmt");
  case CXCursor_NullStmt:
      return createCXString("NullStmt");
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
  case CXCursor_CXXFinalAttr:
      return createCXString("attribute(final)");
  case CXCursor_CXXOverrideAttr:
      return createCXString("attribute(override)");
  case CXCursor_AnnotateAttr:
    return createCXString("attribute(annotate)");
  case CXCursor_AsmLabelAttr:
    return createCXString("asm label");
  case CXCursor_PreprocessingDirective:
    return createCXString("preprocessing directive");
  case CXCursor_MacroDefinition:
    return createCXString("macro definition");
  case CXCursor_MacroExpansion:
    return createCXString("macro expansion");
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
  case CXCursor_TypeAliasDecl:
    return createCXString("TypeAliasDecl");
  case CXCursor_ObjCSynthesizeDecl:
    return createCXString("ObjCSynthesizeDecl");
  case CXCursor_ObjCDynamicDecl:
    return createCXString("ObjCDynamicDecl");
  case CXCursor_CXXAccessSpecifier:
    return createCXString("CXXAccessSpecifier");
  case CXCursor_ModuleImportDecl:
    return createCXString("ModuleImport");
  }

  llvm_unreachable("Unhandled CXCursorKind");
}

struct GetCursorData {
  SourceLocation TokenBeginLoc;
  bool PointsAtMacroArgExpansion;
  bool VisitedObjCPropertyImplDecl;
  SourceLocation VisitedDeclaratorDeclStartLoc;
  CXCursor &BestCursor;

  GetCursorData(SourceManager &SM,
                SourceLocation tokenBegin, CXCursor &outputCursor)
    : TokenBeginLoc(tokenBegin), BestCursor(outputCursor) {
    PointsAtMacroArgExpansion = SM.isMacroArgExpansion(tokenBegin);
    VisitedObjCPropertyImplDecl = false;
  }
};

static enum CXChildVisitResult GetCursorVisitor(CXCursor cursor,
                                                CXCursor parent,
                                                CXClientData client_data) {
  GetCursorData *Data = static_cast<GetCursorData *>(client_data);
  CXCursor *BestCursor = &Data->BestCursor;

  // If we point inside a macro argument we should provide info of what the
  // token is so use the actual cursor, don't replace it with a macro expansion
  // cursor.
  if (cursor.kind == CXCursor_MacroExpansion && Data->PointsAtMacroArgExpansion)
    return CXChildVisit_Recurse;
  
  if (clang_isDeclaration(cursor.kind)) {
    // Avoid having the implicit methods override the property decls.
    if (ObjCMethodDecl *MD
          = dyn_cast_or_null<ObjCMethodDecl>(getCursorDecl(cursor))) {
      if (MD->isImplicit())
        return CXChildVisit_Break;

    } else if (ObjCInterfaceDecl *ID
                 = dyn_cast_or_null<ObjCInterfaceDecl>(getCursorDecl(cursor))) {
      // Check that when we have multiple @class references in the same line,
      // that later ones do not override the previous ones.
      // If we have:
      // @class Foo, Bar;
      // source ranges for both start at '@', so 'Bar' will end up overriding
      // 'Foo' even though the cursor location was at 'Foo'.
      if (BestCursor->kind == CXCursor_ObjCInterfaceDecl ||
          BestCursor->kind == CXCursor_ObjCClassRef)
        if (ObjCInterfaceDecl *PrevID
             = dyn_cast_or_null<ObjCInterfaceDecl>(getCursorDecl(*BestCursor))){
         if (PrevID != ID &&
             !PrevID->isThisDeclarationADefinition() &&
             !ID->isThisDeclarationADefinition())
           return CXChildVisit_Break;
        }

    } else if (DeclaratorDecl *DD
                    = dyn_cast_or_null<DeclaratorDecl>(getCursorDecl(cursor))) {
      SourceLocation StartLoc = DD->getSourceRange().getBegin();
      // Check that when we have multiple declarators in the same line,
      // that later ones do not override the previous ones.
      // If we have:
      // int Foo, Bar;
      // source ranges for both start at 'int', so 'Bar' will end up overriding
      // 'Foo' even though the cursor location was at 'Foo'.
      if (Data->VisitedDeclaratorDeclStartLoc == StartLoc)
        return CXChildVisit_Break;
      Data->VisitedDeclaratorDeclStartLoc = StartLoc;

    } else if (ObjCPropertyImplDecl *PropImp
              = dyn_cast_or_null<ObjCPropertyImplDecl>(getCursorDecl(cursor))) {
      (void)PropImp;
      // Check that when we have multiple @synthesize in the same line,
      // that later ones do not override the previous ones.
      // If we have:
      // @synthesize Foo, Bar;
      // source ranges for both start at '@', so 'Bar' will end up overriding
      // 'Foo' even though the cursor location was at 'Foo'.
      if (Data->VisitedObjCPropertyImplDecl)
        return CXChildVisit_Break;
      Data->VisitedObjCPropertyImplDecl = true;
    }
  }

  if (clang_isExpression(cursor.kind) &&
      clang_isDeclaration(BestCursor->kind)) {
    if (Decl *D = getCursorDecl(*BestCursor)) {
      // Avoid having the cursor of an expression replace the declaration cursor
      // when the expression source range overlaps the declaration range.
      // This can happen for C++ constructor expressions whose range generally
      // include the variable declaration, e.g.:
      //  MyCXXClass foo; // Make sure pointing at 'foo' returns a VarDecl cursor.
      if (D->getLocation().isValid() && Data->TokenBeginLoc.isValid() &&
          D->getLocation() == Data->TokenBeginLoc)
        return CXChildVisit_Break;
    }
  }

  // If our current best cursor is the construction of a temporary object, 
  // don't replace that cursor with a type reference, because we want 
  // clang_getCursor() to point at the constructor.
  if (clang_isExpression(BestCursor->kind) &&
      isa<CXXTemporaryObjectExpr>(getCursorExpr(*BestCursor)) &&
      cursor.kind == CXCursor_TypeRef) {
    // Keep the cursor pointing at CXXTemporaryObjectExpr but also mark it
    // as having the actual point on the type reference.
    *BestCursor = getTypeRefedCallExprCursor(*BestCursor);
    return CXChildVisit_Recurse;
  }
  
  *BestCursor = cursor;
  return CXChildVisit_Recurse;
}

CXCursor clang_getCursor(CXTranslationUnit TU, CXSourceLocation Loc) {
  if (!TU)
    return clang_getNullCursor();

  ASTUnit *CXXUnit = static_cast<ASTUnit *>(TU->TUData);
  ASTUnit::ConcurrencyCheck Check(*CXXUnit);

  SourceLocation SLoc = cxloc::translateSourceLocation(Loc);
  CXCursor Result = cxcursor::getCursor(TU, SLoc);

  bool Logging = getenv("LIBCLANG_LOGGING");  
  if (Logging) {
    CXFile SearchFile;
    unsigned SearchLine, SearchColumn;
    CXFile ResultFile;
    unsigned ResultLine, ResultColumn;
    CXString SearchFileName, ResultFileName, KindSpelling, USR;
    const char *IsDef = clang_isCursorDefinition(Result)? " (Definition)" : "";
    CXSourceLocation ResultLoc = clang_getCursorLocation(Result);
    
    clang_getExpansionLocation(Loc, &SearchFile, &SearchLine, &SearchColumn, 0);
    clang_getExpansionLocation(ResultLoc, &ResultFile, &ResultLine,
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
      clang_getExpansionLocation(DefinitionLoc, &DefinitionFile,
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
  return (K >= CXCursor_FirstDecl && K <= CXCursor_LastDecl) ||
         (K >= CXCursor_FirstExtraDecl && K <= CXCursor_LastExtraDecl);
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

unsigned clang_isAttribute(enum CXCursorKind K) {
    return K >= CXCursor_FirstAttr && K <= CXCursor_LastAttr;
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

    case CXCursor_VariableRef: {
      std::pair<VarDecl *, SourceLocation> P = getCursorVariableRef(C);
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
                                        BaseSpec->getLocStart());
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

  if (C.kind == CXCursor_MacroExpansion) {
    SourceLocation L
      = cxcursor::getCursorMacroExpansion(C)->getSourceRange().getBegin();
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

  if (!clang_isDeclaration(C.kind))
    return clang_getNullLocation();

  Decl *D = getCursorDecl(C);
  if (!D)
    return clang_getNullLocation();

  SourceLocation Loc = D->getLocation();
  // FIXME: Multiple variables declared in a single declaration
  // currently lack the information needed to correctly determine their
  // ranges when accounting for the type-specifier.  We use context
  // stored in the CXCursor to determine if the VarDecl is in a DeclGroup,
  // and if so, whether it is the first decl.
  if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
    if (!cxcursor::isFirstInDeclGroup(C))
      Loc = VD->getLocation();
  }

  // For ObjC methods, give the start location of the method name.
  if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
    Loc = MD->getSelectorStartLoc();

  return cxloc::translateSourceLocation(getCursorContext(C), Loc);
}

} // end extern "C"

CXCursor cxcursor::getCursor(CXTranslationUnit TU, SourceLocation SLoc) {
  assert(TU);

  // Guard against an invalid SourceLocation, or we may assert in one
  // of the following calls.
  if (SLoc.isInvalid())
    return clang_getNullCursor();

  ASTUnit *CXXUnit = static_cast<ASTUnit *>(TU->TUData);

  // Translate the given source location to make it point at the beginning of
  // the token under the cursor.
  SLoc = Lexer::GetBeginningOfToken(SLoc, CXXUnit->getSourceManager(),
                                    CXXUnit->getASTContext().getLangOpts());
  
  CXCursor Result = MakeCXCursorInvalid(CXCursor_NoDeclFound);
  if (SLoc.isValid()) {
    GetCursorData ResultData(CXXUnit->getSourceManager(), SLoc, Result);
    CursorVisitor CursorVis(TU, GetCursorVisitor, &ResultData,
                            /*VisitPreprocessorLast=*/true, 
                            /*VisitIncludedEntities=*/false,
                            SourceLocation(SLoc));
    CursorVis.visitFileRegion();
  }

  return Result;
}

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

    case CXCursor_VariableRef:
      return getCursorVariableRef(C).second;
        
    default:
      // FIXME: Need a way to enumerate all non-reference cases.
      llvm_unreachable("Missed a reference kind");
    }
  }

  if (clang_isExpression(C.kind))
    return getCursorExpr(C)->getSourceRange();

  if (clang_isStatement(C.kind))
    return getCursorStmt(C)->getSourceRange();

  if (clang_isAttribute(C.kind))
    return getCursorAttr(C)->getRange();

  if (C.kind == CXCursor_PreprocessingDirective)
    return cxcursor::getCursorPreprocessingDirective(C);

  if (C.kind == CXCursor_MacroExpansion) {
    ASTUnit *TU = getCursorASTUnit(C);
    SourceRange Range = cxcursor::getCursorMacroExpansion(C)->getSourceRange();
    return TU->mapRangeFromPreamble(Range);
  }

  if (C.kind == CXCursor_MacroDefinition) {
    ASTUnit *TU = getCursorASTUnit(C);
    SourceRange Range = cxcursor::getCursorMacroDefinition(C)->getSourceRange();
    return TU->mapRangeFromPreamble(Range);
  }

  if (C.kind == CXCursor_InclusionDirective) {
    ASTUnit *TU = getCursorASTUnit(C);
    SourceRange Range = cxcursor::getCursorInclusionDirective(C)->getSourceRange();
    return TU->mapRangeFromPreamble(Range);
  }

  if (C.kind == CXCursor_TranslationUnit) {
    ASTUnit *TU = getCursorASTUnit(C);
    FileID MainID = TU->getSourceManager().getMainFileID();
    SourceLocation Start = TU->getSourceManager().getLocForStartOfFile(MainID);
    SourceLocation End = TU->getSourceManager().getLocForEndOfFile(MainID);
    return SourceRange(Start, End);
  }

  if (clang_isDeclaration(C.kind)) {
    Decl *D = cxcursor::getCursorDecl(C);
    if (!D)
      return SourceRange();

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
  if (clang_isDeclaration(C.kind)) {
    Decl *D = cxcursor::getCursorDecl(C);
    if (!D)
      return SourceRange();

    SourceRange R = D->getSourceRange();

    // Adjust the start of the location for declarations preceded by
    // declaration specifiers.
    SourceLocation StartLoc;
    if (const DeclaratorDecl *DD = dyn_cast<DeclaratorDecl>(D)) {
      if (TypeSourceInfo *TI = DD->getTypeSourceInfo())
        StartLoc = TI->getTypeLoc().getLocStart();
    } else if (TypedefDecl *Typedef = dyn_cast<TypedefDecl>(D)) {
      if (TypeSourceInfo *TI = Typedef->getTypeSourceInfo())
        StartLoc = TI->getTypeLoc().getLocStart();
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
    if (!D)
      return clang_getNullCursor();
    if (UsingDecl *Using = dyn_cast<UsingDecl>(D))
      return MakeCursorOverloadedDeclRef(Using, D->getLocation(), tu);
    if (ObjCPropertyImplDecl *PropImpl =dyn_cast<ObjCPropertyImplDecl>(D))
      if (ObjCPropertyDecl *Property = PropImpl->getPropertyDecl())
        return MakeCXCursor(Property, tu);
    
    return C;
  }
  
  if (clang_isExpression(C.kind)) {
    Expr *E = getCursorExpr(C);
    Decl *D = getDeclFromExpr(E);
    if (D) {
      CXCursor declCursor = MakeCXCursor(D, tu);
      declCursor = getSelectorIdentifierCursor(getSelectorIdentifierIndex(C),
                                               declCursor);
      return declCursor;
    }
    
    if (OverloadExpr *Ovl = dyn_cast_or_null<OverloadExpr>(E))
      return MakeCursorOverloadedDeclRef(Ovl, tu);
        
    return clang_getNullCursor();
  }

  if (clang_isStatement(C.kind)) {
    Stmt *S = getCursorStmt(C);
    if (GotoStmt *Goto = dyn_cast_or_null<GotoStmt>(S))
      if (LabelDecl *label = Goto->getLabel())
        if (LabelStmt *labelS = label->getStmt())
        return MakeCXCursor(labelS, getCursorDecl(C), tu);

    return clang_getNullCursor();
  }
  
  if (C.kind == CXCursor_MacroExpansion) {
    if (MacroDefinition *Def = getCursorMacroExpansion(C)->getDefinition())
      return MakeMacroDefinitionCursor(Def, tu);
  }

  if (!clang_isReference(C.kind))
    return clang_getNullCursor();

  switch (C.kind) {
    case CXCursor_ObjCSuperClassRef:
      return MakeCXCursor(getCursorObjCSuperClassRef(C).first, tu);

    case CXCursor_ObjCProtocolRef: {
      ObjCProtocolDecl *Prot = getCursorObjCProtocolRef(C).first;
      if (ObjCProtocolDecl *Def = Prot->getDefinition())
        return MakeCXCursor(Def, tu);

      return MakeCXCursor(Prot, tu);
    }

    case CXCursor_ObjCClassRef: {
      ObjCInterfaceDecl *Class = getCursorObjCClassRef(C).first;
      if (ObjCInterfaceDecl *Def = Class->getDefinition())
        return MakeCXCursor(Def, tu);

      return MakeCXCursor(Class, tu);
    }

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
      
    case CXCursor_VariableRef:
      return MakeCXCursor(getCursorVariableRef(C).first, tu);

    default:
      // We would prefer to enumerate all non-reference cursor kinds here.
      llvm_unreachable("Unhandled reference cursor kind");
  }
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

  if (C.kind == CXCursor_MacroExpansion)
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
  case Decl::TypeAlias:
  case Decl::TypeAliasTemplate:
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
  case Decl::ClassScopeFunctionSpecialization:
  case Decl::Import:
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
    if (ObjCProtocolDecl *Def = cast<ObjCProtocolDecl>(D)->getDefinition())
      return MakeCXCursor(Def, TU);
    return clang_getNullCursor();

  case Decl::ObjCInterface: {
    // There are two notions of a "definition" for an Objective-C
    // class: the interface and its implementation. When we resolved a
    // reference to an Objective-C class, produce the @interface as
    // the definition; when we were provided with the interface,
    // produce the @implementation as the definition.
    ObjCInterfaceDecl *IFace = cast<ObjCInterfaceDecl>(D);
    if (WasReference) {
      if (ObjCInterfaceDecl *Def = IFace->getDefinition())
        return MakeCXCursor(Def, TU);
    } else if (ObjCImplementationDecl *Impl = IFace->getImplementation())
      return MakeCXCursor(Impl, TU);
    return clang_getNullCursor();
  }

  case Decl::ObjCProperty:
    // FIXME: We don't really know where to find the
    // ObjCPropertyImplDecls that implement this property.
    return clang_getNullCursor();

  case Decl::ObjCCompatibleAlias:
    if (ObjCInterfaceDecl *Class
          = cast<ObjCCompatibleAliasDecl>(D)->getClassInterface())
      if (ObjCInterfaceDecl *Def = Class->getDefinition())
        return MakeCXCursor(Def, TU);

    return clang_getNullCursor();

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
  
  if (Decl *D = getCursorDecl(C)) {
    if (ObjCCategoryImplDecl *CatImplD = dyn_cast<ObjCCategoryImplDecl>(D))
      if (ObjCCategoryDecl *CatD = CatImplD->getCategoryDecl())
        return MakeCXCursor(CatD, getCursorTU(C));

    if (ObjCImplDecl *ImplD = dyn_cast<ObjCImplDecl>(D))
      if (ObjCInterfaceDecl *IFD = ImplD->getClassInterface())
        return MakeCXCursor(IFD, getCursorTU(C));

    return MakeCXCursor(D->getCanonicalDecl(), getCursorTU(C));
  }
  
  return C;
}

int clang_Cursor_getObjCSelectorIndex(CXCursor cursor) {
  return cxcursor::getSelectorIdentifierIndexAndLoc(cursor).first;
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


CXSourceRange clang_getCursorReferenceNameRange(CXCursor C, unsigned NameFlags,
                                                unsigned PieceIndex) {
  RefNamePieces Pieces;
  
  switch (C.kind) {
  case CXCursor_MemberRefExpr:
    if (MemberExpr *E = dyn_cast<MemberExpr>(getCursorExpr(C)))
      Pieces = buildPieces(NameFlags, true, E->getMemberNameInfo(),
                           E->getQualifierLoc().getSourceRange());
    break;
  
  case CXCursor_DeclRefExpr:
    if (DeclRefExpr *E = dyn_cast<DeclRefExpr>(getCursorExpr(C)))
      Pieces = buildPieces(NameFlags, false, E->getNameInfo(), 
                           E->getQualifierLoc().getSourceRange(),
                           E->getOptionalExplicitTemplateArgs());
    break;
    
  case CXCursor_CallExpr:
    if (CXXOperatorCallExpr *OCE = 
        dyn_cast<CXXOperatorCallExpr>(getCursorExpr(C))) {
      Expr *Callee = OCE->getCallee();
      if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(Callee))
        Callee = ICE->getSubExpr();

      if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Callee))
        Pieces = buildPieces(NameFlags, false, DRE->getNameInfo(),
                             DRE->getQualifierLoc().getSourceRange());
    }
    break;
    
  default:
    break;
  }

  if (Pieces.empty()) {
    if (PieceIndex == 0)
      return clang_getCursorExtent(C);
  } else if (PieceIndex < Pieces.size()) {
      SourceRange R = Pieces[PieceIndex];
      if (R.isValid())
        return cxloc::translateSourceRange(getCursorContext(C), R);
  }
  
  return clang_getNullRange();
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
    return createCXString(StringRef(Text, CXTok.int_data[2]));
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
    = CXXUnit->getSourceManager().getDecomposedSpellingLoc(Loc);
  bool Invalid = false;
  StringRef Buffer
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

static void getTokens(ASTUnit *CXXUnit, SourceRange Range,
                      SmallVectorImpl<CXToken> &CXTokens) {
  SourceManager &SourceMgr = CXXUnit->getSourceManager();
  std::pair<FileID, unsigned> BeginLocInfo
    = SourceMgr.getDecomposedLoc(Range.getBegin());
  std::pair<FileID, unsigned> EndLocInfo
    = SourceMgr.getDecomposedLoc(Range.getEnd());

  // Cannot tokenize across files.
  if (BeginLocInfo.first != EndLocInfo.first)
    return;

  // Create a lexer
  bool Invalid = false;
  StringRef Buffer
    = SourceMgr.getBufferData(BeginLocInfo.first, &Invalid);
  if (Invalid)
    return;
  
  Lexer Lex(SourceMgr.getLocForStartOfFile(BeginLocInfo.first),
            CXXUnit->getASTContext().getLangOpts(),
            Buffer.begin(), Buffer.data() + BeginLocInfo.second, Buffer.end());
  Lex.SetCommentRetentionState(true);

  // Lex tokens until we hit the end of the range.
  const char *EffectiveBufferEnd = Buffer.data() + EndLocInfo.second;
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

  SmallVector<CXToken, 32> CXTokens;
  getTokens(CXXUnit, R, CXTokens);

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
static bool AnnotateTokensPostChildrenVisitor(CXCursor cursor,
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
  bool HasContextSensitiveKeywords;

  struct PostChildrenInfo {
    CXCursor Cursor;
    SourceRange CursorRange;
    unsigned BeforeChildrenTokenIdx;
  };
  llvm::SmallVector<PostChildrenInfo, 8> PostChildrenInfos;
  
  bool MoreTokens() const { return TokIdx < NumTokens; }
  unsigned NextToken() const { return TokIdx; }
  void AdvanceToken() { ++TokIdx; }
  SourceLocation GetTokenLoc(unsigned tokI) {
    return SourceLocation::getFromRawEncoding(Tokens[tokI].int_data[1]);
  }
  bool isFunctionMacroToken(unsigned tokI) const {
    return Tokens[tokI].int_data[3] != 0;
  }
  SourceLocation getFunctionMacroTokenLoc(unsigned tokI) const {
    return SourceLocation::getFromRawEncoding(Tokens[tokI].int_data[3]);
  }

  void annotateAndAdvanceTokens(CXCursor, RangeComparisonResult, SourceRange);
  void annotateAndAdvanceFunctionMacroTokens(CXCursor, RangeComparisonResult,
                                             SourceRange);

public:
  AnnotateTokensWorker(AnnotateTokensData &annotated,
                       CXToken *tokens, CXCursor *cursors, unsigned numTokens,
                       CXTranslationUnit tu, SourceRange RegionOfInterest)
    : Annotated(annotated), Tokens(tokens), Cursors(cursors),
      NumTokens(numTokens), TokIdx(0), PreprocessingTokIdx(0),
      AnnotateVis(tu,
                  AnnotateTokensVisitor, this,
                  /*VisitPreprocessorLast=*/true,
                  /*VisitIncludedEntities=*/false,
                  RegionOfInterest,
                  /*VisitDeclsOnly=*/false,
                  AnnotateTokensPostChildrenVisitor),
      SrcMgr(static_cast<ASTUnit*>(tu->TUData)->getSourceManager()),
      HasContextSensitiveKeywords(false) { }

  void VisitChildren(CXCursor C) { AnnotateVis.VisitChildren(C); }
  enum CXChildVisitResult Visit(CXCursor cursor, CXCursor parent);
  bool postVisitChildren(CXCursor cursor);
  void AnnotateTokens();
  
  /// \brief Determine whether the annotator saw any cursors that have 
  /// context-sensitive keywords.
  bool hasContextSensitiveKeywords() const {
    return HasContextSensitiveKeywords;
  }

  ~AnnotateTokensWorker() {
    assert(PostChildrenInfos.empty());
  }
};
}

void AnnotateTokensWorker::AnnotateTokens() {
  // Walk the AST within the region of interest, annotating tokens
  // along the way.
  AnnotateVis.visitFileRegion();

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
    if (I < PreprocessingTokIdx && clang_isPreprocessing(Cursors[I].kind))
      continue;

    AnnotateTokensData::iterator Pos = Annotated.find(Tokens[I].int_data[1]);
    Cursors[I] = (Pos == Annotated.end()) ? C : Pos->second;
  }
}

/// \brief It annotates and advances tokens with a cursor until the comparison
//// between the cursor location and the source range is the same as
/// \arg compResult.
///
/// Pass RangeBefore to annotate tokens with a cursor until a range is reached.
/// Pass RangeOverlap to annotate tokens inside a range.
void AnnotateTokensWorker::annotateAndAdvanceTokens(CXCursor updateC,
                                               RangeComparisonResult compResult,
                                               SourceRange range) {
  while (MoreTokens()) {
    const unsigned I = NextToken();
    if (isFunctionMacroToken(I))
      return annotateAndAdvanceFunctionMacroTokens(updateC, compResult, range);

    SourceLocation TokLoc = GetTokenLoc(I);
    if (LocationCompare(SrcMgr, TokLoc, range) == compResult) {
      Cursors[I] = updateC;
      AdvanceToken();
      continue;
    }
    break;
  }
}

/// \brief Special annotation handling for macro argument tokens.
void AnnotateTokensWorker::annotateAndAdvanceFunctionMacroTokens(
                                               CXCursor updateC,
                                               RangeComparisonResult compResult,
                                               SourceRange range) {
  assert(MoreTokens());
  assert(isFunctionMacroToken(NextToken()) &&
         "Should be called only for macro arg tokens");

  // This works differently than annotateAndAdvanceTokens; because expanded
  // macro arguments can have arbitrary translation-unit source order, we do not
  // advance the token index one by one until a token fails the range test.
  // We only advance once past all of the macro arg tokens if all of them
  // pass the range test. If one of them fails we keep the token index pointing
  // at the start of the macro arg tokens so that the failing token will be
  // annotated by a subsequent annotation try.

  bool atLeastOneCompFail = false;
  
  unsigned I = NextToken();
  for (; I < NumTokens && isFunctionMacroToken(I); ++I) {
    SourceLocation TokLoc = getFunctionMacroTokenLoc(I);
    if (TokLoc.isFileID())
      continue; // not macro arg token, it's parens or comma.
    if (LocationCompare(SrcMgr, TokLoc, range) == compResult) {
      if (clang_isInvalid(clang_getCursorKind(Cursors[I])))
        Cursors[I] = updateC;
    } else
      atLeastOneCompFail = true;
  }

  if (!atLeastOneCompFail)
    TokIdx = I; // All of the tokens were handled, advance beyond all of them.
}

enum CXChildVisitResult
AnnotateTokensWorker::Visit(CXCursor cursor, CXCursor parent) {  
  CXSourceLocation Loc = clang_getCursorLocation(cursor);
  SourceRange cursorRange = getRawCursorExtent(cursor);
  if (cursorRange.isInvalid())
    return CXChildVisit_Recurse;
      
  if (!HasContextSensitiveKeywords) {
    // Objective-C properties can have context-sensitive keywords.
    if (cursor.kind == CXCursor_ObjCPropertyDecl) {
      if (ObjCPropertyDecl *Property 
                  = dyn_cast_or_null<ObjCPropertyDecl>(getCursorDecl(cursor)))
        HasContextSensitiveKeywords = Property->getPropertyAttributesAsWritten() != 0;
    }
    // Objective-C methods can have context-sensitive keywords.
    else if (cursor.kind == CXCursor_ObjCInstanceMethodDecl ||
             cursor.kind == CXCursor_ObjCClassMethodDecl) {
      if (ObjCMethodDecl *Method
            = dyn_cast_or_null<ObjCMethodDecl>(getCursorDecl(cursor))) {
        if (Method->getObjCDeclQualifier())
          HasContextSensitiveKeywords = true;
        else {
          for (ObjCMethodDecl::param_iterator P = Method->param_begin(),
                                           PEnd = Method->param_end();
               P != PEnd; ++P) {
            if ((*P)->getObjCDeclQualifier()) {
              HasContextSensitiveKeywords = true;
              break;
            }
          }
        }
      }
    }    
    // C++ methods can have context-sensitive keywords.
    else if (cursor.kind == CXCursor_CXXMethod) {
      if (CXXMethodDecl *Method
                  = dyn_cast_or_null<CXXMethodDecl>(getCursorDecl(cursor))) {
        if (Method->hasAttr<FinalAttr>() || Method->hasAttr<OverrideAttr>())
          HasContextSensitiveKeywords = true;
      }
    }
    // C++ classes can have context-sensitive keywords.
    else if (cursor.kind == CXCursor_StructDecl ||
             cursor.kind == CXCursor_ClassDecl ||
             cursor.kind == CXCursor_ClassTemplate ||
             cursor.kind == CXCursor_ClassTemplatePartialSpecialization) {
      if (Decl *D = getCursorDecl(cursor))
        if (D->hasAttr<FinalAttr>())
          HasContextSensitiveKeywords = true;
    }
  }
  
  if (clang_isPreprocessing(cursor.kind)) {    
    // For macro expansions, just note where the beginning of the macro
    // expansion occurs.
    if (cursor.kind == CXCursor_MacroExpansion) {
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
        llvm_unreachable("Infeasible");
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
  if (clang_isDeclaration(cursorK)) {
    Decl *D = cxcursor::getCursorDecl(cursor);
    
    SourceLocation StartLoc;
    if (const DeclaratorDecl *DD = dyn_cast_or_null<DeclaratorDecl>(D)) {
      if (TypeSourceInfo *TI = DD->getTypeSourceInfo())
        StartLoc = TI->getTypeLoc().getLocStart();
    } else if (TypedefDecl *Typedef = dyn_cast_or_null<TypedefDecl>(D)) {
      if (TypeSourceInfo *TI = Typedef->getTypeSourceInfo())
        StartLoc = TI->getTypeLoc().getLocStart();
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

  annotateAndAdvanceTokens(updateC, RangeBefore, cursorRange);

  // Avoid having the cursor of an expression "overwrite" the annotation of the
  // variable declaration that it belongs to.
  // This can happen for C++ constructor expressions whose range generally
  // include the variable declaration, e.g.:
  //  MyCXXClass foo; // Make sure we don't annotate 'foo' as a CallExpr cursor.
  if (clang_isExpression(cursorK)) {
    Expr *E = getCursorExpr(cursor);
    if (Decl *D = getCursorParentDecl(cursor)) {
      const unsigned I = NextToken();
      if (E->getLocStart().isValid() && D->getLocation().isValid() &&
          E->getLocStart() == D->getLocation() &&
          E->getLocStart() == GetTokenLoc(I)) {
        Cursors[I] = updateC;
        AdvanceToken();
      }
    }
  }

  // Before recursing into the children keep some state that we are going
  // to use in the AnnotateTokensWorker::postVisitChildren callback to do some
  // extra work after the child nodes are visited.
  // Note that we don't call VisitChildren here to avoid traversing statements
  // code-recursively which can blow the stack.

  PostChildrenInfo Info;
  Info.Cursor = cursor;
  Info.CursorRange = cursorRange;
  Info.BeforeChildrenTokenIdx = NextToken();
  PostChildrenInfos.push_back(Info);

  return CXChildVisit_Recurse;
}

bool AnnotateTokensWorker::postVisitChildren(CXCursor cursor) {
  if (PostChildrenInfos.empty())
    return false;
  const PostChildrenInfo &Info = PostChildrenInfos.back();
  if (!clang_equalCursors(Info.Cursor, cursor))
    return false;

  const unsigned BeforeChildren = Info.BeforeChildrenTokenIdx;
  const unsigned AfterChildren = NextToken();
  SourceRange cursorRange = Info.CursorRange;

  // Scan the tokens that are at the end of the cursor, but are not captured
  // but the child cursors.
  annotateAndAdvanceTokens(cursor, RangeOverlap, cursorRange);

  // Scan the tokens that are at the beginning of the cursor, but are not
  // capture by the child cursors.
  for (unsigned I = BeforeChildren; I != AfterChildren; ++I) {
    if (!clang_isInvalid(clang_getCursorKind(Cursors[I])))
      break;

    Cursors[I] = cursor;
  }

  PostChildrenInfos.pop_back();
  return false;
}

static enum CXChildVisitResult AnnotateTokensVisitor(CXCursor cursor,
                                                     CXCursor parent,
                                                     CXClientData client_data) {
  return static_cast<AnnotateTokensWorker*>(client_data)->Visit(cursor, parent);
}

static bool AnnotateTokensPostChildrenVisitor(CXCursor cursor,
                                              CXClientData client_data) {
  return static_cast<AnnotateTokensWorker*>(client_data)->
                                                      postVisitChildren(cursor);
}

namespace {

/// \brief Uses the macro expansions in the preprocessing record to find
/// and mark tokens that are macro arguments. This info is used by the
/// AnnotateTokensWorker.
class MarkMacroArgTokensVisitor {
  SourceManager &SM;
  CXToken *Tokens;
  unsigned NumTokens;
  unsigned CurIdx;
  
public:
  MarkMacroArgTokensVisitor(SourceManager &SM,
                            CXToken *tokens, unsigned numTokens)
    : SM(SM), Tokens(tokens), NumTokens(numTokens), CurIdx(0) { }

  CXChildVisitResult visit(CXCursor cursor, CXCursor parent) {
    if (cursor.kind != CXCursor_MacroExpansion)
      return CXChildVisit_Continue;

    SourceRange macroRange = getCursorMacroExpansion(cursor)->getSourceRange();
    if (macroRange.getBegin() == macroRange.getEnd())
      return CXChildVisit_Continue; // it's not a function macro.

    for (; CurIdx < NumTokens; ++CurIdx) {
      if (!SM.isBeforeInTranslationUnit(getTokenLoc(CurIdx),
                                        macroRange.getBegin()))
        break;
    }
    
    if (CurIdx == NumTokens)
      return CXChildVisit_Break;

    for (; CurIdx < NumTokens; ++CurIdx) {
      SourceLocation tokLoc = getTokenLoc(CurIdx);
      if (!SM.isBeforeInTranslationUnit(tokLoc, macroRange.getEnd()))
        break;

      setFunctionMacroTokenLoc(CurIdx, SM.getMacroArgExpandedLocation(tokLoc));
    }

    if (CurIdx == NumTokens)
      return CXChildVisit_Break;

    return CXChildVisit_Continue;
  }

private:
  SourceLocation getTokenLoc(unsigned tokI) {
    return SourceLocation::getFromRawEncoding(Tokens[tokI].int_data[1]);
  }

  void setFunctionMacroTokenLoc(unsigned tokI, SourceLocation loc) {
    // The third field is reserved and currently not used. Use it here
    // to mark macro arg expanded tokens with their expanded locations.
    Tokens[tokI].int_data[3] = loc.getRawEncoding();
  }
};

} // end anonymous namespace

static CXChildVisitResult
MarkMacroArgTokensVisitorDelegate(CXCursor cursor, CXCursor parent,
                                  CXClientData client_data) {
  return static_cast<MarkMacroArgTokensVisitor*>(client_data)->visit(cursor,
                                                                     parent);
}

namespace {
  struct clang_annotateTokens_Data {
    CXTranslationUnit TU;
    ASTUnit *CXXUnit;
    CXToken *Tokens;
    unsigned NumTokens;
    CXCursor *Cursors;
  };
}

static void annotatePreprocessorTokens(CXTranslationUnit TU,
                                       SourceRange RegionOfInterest,
                                       AnnotateTokensData &Annotated) {
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(TU->TUData);

  SourceManager &SourceMgr = CXXUnit->getSourceManager();
  std::pair<FileID, unsigned> BeginLocInfo
    = SourceMgr.getDecomposedLoc(RegionOfInterest.getBegin());
  std::pair<FileID, unsigned> EndLocInfo
    = SourceMgr.getDecomposedLoc(RegionOfInterest.getEnd());

  if (BeginLocInfo.first != EndLocInfo.first)
    return;

  StringRef Buffer;
  bool Invalid = false;
  Buffer = SourceMgr.getBufferData(BeginLocInfo.first, &Invalid);
  if (Buffer.empty() || Invalid)
    return;

  Lexer Lex(SourceMgr.getLocForStartOfFile(BeginLocInfo.first),
            CXXUnit->getASTContext().getLangOpts(),
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
      SmallVector<SourceLocation, 32> Locations;
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

// This gets run a separate thread to avoid stack blowout.
static void clang_annotateTokensImpl(void *UserData) {
  CXTranslationUnit TU = ((clang_annotateTokens_Data*)UserData)->TU;
  ASTUnit *CXXUnit = ((clang_annotateTokens_Data*)UserData)->CXXUnit;
  CXToken *Tokens = ((clang_annotateTokens_Data*)UserData)->Tokens;
  const unsigned NumTokens = ((clang_annotateTokens_Data*)UserData)->NumTokens;
  CXCursor *Cursors = ((clang_annotateTokens_Data*)UserData)->Cursors;

  CIndexer *CXXIdx = (CIndexer*)TU->CIdx;
  if (CXXIdx->isOptEnabled(CXGlobalOpt_ThreadBackgroundPriorityForEditing))
    setThreadBackgroundPriority();

  // Determine the region of interest, which contains all of the tokens.
  SourceRange RegionOfInterest;
  RegionOfInterest.setBegin(
    cxloc::translateSourceLocation(clang_getTokenLocation(TU, Tokens[0])));
  RegionOfInterest.setEnd(
    cxloc::translateSourceLocation(clang_getTokenLocation(TU,
                                                         Tokens[NumTokens-1])));

  // A mapping from the source locations found when re-lexing or traversing the
  // region of interest to the corresponding cursors.
  AnnotateTokensData Annotated;

  // Relex the tokens within the source range to look for preprocessing
  // directives.
  annotatePreprocessorTokens(TU, RegionOfInterest, Annotated);
  
  if (CXXUnit->getPreprocessor().getPreprocessingRecord()) {
    // Search and mark tokens that are macro argument expansions.
    MarkMacroArgTokensVisitor Visitor(CXXUnit->getSourceManager(),
                                      Tokens, NumTokens);
    CursorVisitor MacroArgMarker(TU,
                                 MarkMacroArgTokensVisitorDelegate, &Visitor,
                                 /*VisitPreprocessorLast=*/true,
                                 /*VisitIncludedEntities=*/false,
                                 RegionOfInterest);
    MacroArgMarker.visitPreprocessedEntitiesInRegion();
  }
  
  // Annotate all of the source locations in the region of interest that map to
  // a specific cursor.
  AnnotateTokensWorker W(Annotated, Tokens, Cursors, NumTokens,
                         TU, RegionOfInterest);
  
  // FIXME: We use a ridiculous stack size here because the data-recursion
  // algorithm uses a large stack frame than the non-data recursive version,
  // and AnnotationTokensWorker currently transforms the data-recursion
  // algorithm back into a traditional recursion by explicitly calling
  // VisitChildren().  We will need to remove this explicit recursive call.
  W.AnnotateTokens();

  // If we ran into any entities that involve context-sensitive keywords,
  // take another pass through the tokens to mark them as such.
  if (W.hasContextSensitiveKeywords()) {
    for (unsigned I = 0; I != NumTokens; ++I) {
      if (clang_getTokenKind(Tokens[I]) != CXToken_Identifier)
        continue;
      
      if (Cursors[I].kind == CXCursor_ObjCPropertyDecl) {
        IdentifierInfo *II = static_cast<IdentifierInfo *>(Tokens[I].ptr_data);
        if (ObjCPropertyDecl *Property
            = dyn_cast_or_null<ObjCPropertyDecl>(getCursorDecl(Cursors[I]))) {
          if (Property->getPropertyAttributesAsWritten() != 0 &&
              llvm::StringSwitch<bool>(II->getName())
              .Case("readonly", true)
              .Case("assign", true)
              .Case("unsafe_unretained", true)
              .Case("readwrite", true)
              .Case("retain", true)
              .Case("copy", true)
              .Case("nonatomic", true)
              .Case("atomic", true)
              .Case("getter", true)
              .Case("setter", true)
              .Case("strong", true)
              .Case("weak", true)
              .Default(false))
            Tokens[I].int_data[0] = CXToken_Keyword;
        }
        continue;
      }
      
      if (Cursors[I].kind == CXCursor_ObjCInstanceMethodDecl ||
          Cursors[I].kind == CXCursor_ObjCClassMethodDecl) {
        IdentifierInfo *II = static_cast<IdentifierInfo *>(Tokens[I].ptr_data);
        if (llvm::StringSwitch<bool>(II->getName())
            .Case("in", true)
            .Case("out", true)
            .Case("inout", true)
            .Case("oneway", true)
            .Case("bycopy", true)
            .Case("byref", true)
            .Default(false))
          Tokens[I].int_data[0] = CXToken_Keyword;
        continue;
      }

      if (Cursors[I].kind == CXCursor_CXXFinalAttr ||
          Cursors[I].kind == CXCursor_CXXOverrideAttr) {
        Tokens[I].int_data[0] = CXToken_Keyword;
        continue;
      }
    }
  }
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
  
  clang_annotateTokens_Data data = { TU, CXXUnit, Tokens, NumTokens, Cursors };
  llvm::CrashRecoveryContext CRC;
  if (!RunSafely(CRC, clang_annotateTokensImpl, &data,
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
  if (!D)
    return CXLanguage_C;

  switch (D->getKind()) {
    default:
      break;
    case Decl::ImplicitParam:
    case Decl::ObjCAtDefsField:
    case Decl::ObjCCategory:
    case Decl::ObjCCategoryImpl:
    case Decl::ObjCCompatibleAlias:
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
      if (isa<FunctionDecl>(D) && cast<FunctionDecl>(D)->isDeleted())
        return CXAvailability_Available;
      
      switch (D->getAvailability()) {
      case AR_Available:
      case AR_NotYetIntroduced:
        return CXAvailability_Available;

      case AR_Deprecated:
        return CXAvailability_Deprecated;

      case AR_Unavailable:
        return CXAvailability_NotAvailable;
      }
    }

  return CXAvailability_Available;
}

static CXVersion convertVersion(VersionTuple In) {
  CXVersion Out = { -1, -1, -1 };
  if (In.empty())
    return Out;

  Out.Major = In.getMajor();
  
  if (llvm::Optional<unsigned> Minor = In.getMinor())
    Out.Minor = *Minor;
  else
    return Out;

  if (llvm::Optional<unsigned> Subminor = In.getSubminor())
    Out.Subminor = *Subminor;
  
  return Out;
}
  
int clang_getCursorPlatformAvailability(CXCursor cursor,
                                        int *always_deprecated,
                                        CXString *deprecated_message,
                                        int *always_unavailable,
                                        CXString *unavailable_message,
                                        CXPlatformAvailability *availability,
                                        int availability_size) {
  if (always_deprecated)
    *always_deprecated = 0;
  if (deprecated_message)
    *deprecated_message = cxstring::createCXString("", /*DupString=*/false);
  if (always_unavailable)
    *always_unavailable = 0;
  if (unavailable_message)
    *unavailable_message = cxstring::createCXString("", /*DupString=*/false);
  
  if (!clang_isDeclaration(cursor.kind))
    return 0;
  
  Decl *D = cxcursor::getCursorDecl(cursor);
  if (!D)
    return 0;
  
  int N = 0;
  for (Decl::attr_iterator A = D->attr_begin(), AEnd = D->attr_end(); A != AEnd;
       ++A) {
    if (DeprecatedAttr *Deprecated = dyn_cast<DeprecatedAttr>(*A)) {
      if (always_deprecated)
        *always_deprecated = 1;
      if (deprecated_message)
        *deprecated_message = cxstring::createCXString(Deprecated->getMessage());
      continue;
    }
    
    if (UnavailableAttr *Unavailable = dyn_cast<UnavailableAttr>(*A)) {
      if (always_unavailable)
        *always_unavailable = 1;
      if (unavailable_message) {
        *unavailable_message
          = cxstring::createCXString(Unavailable->getMessage());
      }
      continue;
    }
    
    if (AvailabilityAttr *Avail = dyn_cast<AvailabilityAttr>(*A)) {
      if (N < availability_size) {
        availability[N].Platform
          = cxstring::createCXString(Avail->getPlatform()->getName());
        availability[N].Introduced = convertVersion(Avail->getIntroduced());
        availability[N].Deprecated = convertVersion(Avail->getDeprecated());
        availability[N].Obsoleted = convertVersion(Avail->getObsoleted());
        availability[N].Unavailable = Avail->getUnavailable();
        availability[N].Message = cxstring::createCXString(Avail->getMessage());
      }
      ++N;
    }
  }
  
  return N;
}
  
void clang_disposeCXPlatformAvailability(CXPlatformAvailability *availability) {
  clang_disposeString(availability->Platform);
  clang_disposeString(availability->Message);
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

CXFile clang_getIncludedFile(CXCursor cursor) {
  if (cursor.kind != CXCursor_InclusionDirective)
    return 0;
  
  InclusionDirective *ID = getCursorInclusionDirective(cursor);
  return (void *)ID->getFile();
}

CXSourceRange clang_Cursor_getCommentRange(CXCursor C) {
  if (!clang_isDeclaration(C.kind))
    return clang_getNullRange();

  const Decl *D = getCursorDecl(C);
  ASTContext &Context = getCursorContext(C);
  const RawComment *RC = Context.getRawCommentForAnyRedecl(D);
  if (!RC)
    return clang_getNullRange();

  return cxloc::translateSourceRange(Context, RC->getSourceRange());
}

CXString clang_Cursor_getRawCommentText(CXCursor C) {
  if (!clang_isDeclaration(C.kind))
    return createCXString((const char *) NULL);

  const Decl *D = getCursorDecl(C);
  ASTContext &Context = getCursorContext(C);
  const RawComment *RC = Context.getRawCommentForAnyRedecl(D);
  StringRef RawText = RC ? RC->getRawText(Context.getSourceManager()) :
                           StringRef();

  // Don't duplicate the string because RawText points directly into source
  // code.
  return createCXString(RawText, false);
}

CXString clang_Cursor_getBriefCommentText(CXCursor C) {
  if (!clang_isDeclaration(C.kind))
    return createCXString((const char *) NULL);

  const Decl *D = getCursorDecl(C);
  const ASTContext &Context = getCursorContext(C);
  const RawComment *RC = Context.getRawCommentForAnyRedecl(D);

  if (RC) {
    StringRef BriefText = RC->getBriefText(Context);

    // Don't duplicate the string because RawComment ensures that this memory
    // will not go away.
    return createCXString(BriefText, false);
  }

  return createCXString((const char *) NULL);
}

CXComment clang_Cursor_getParsedComment(CXCursor C) {
  if (!clang_isDeclaration(C.kind))
    return cxcomment::createCXComment(NULL, NULL);

  const Decl *D = getCursorDecl(C);
  const ASTContext &Context = getCursorContext(C);
  const comments::FullComment *FC = Context.getCommentForDecl(D, /*PP=*/ NULL);

  return cxcomment::createCXComment(FC, getCursorTU(C));
}

CXModule clang_Cursor_getModule(CXCursor C) {
  if (C.kind == CXCursor_ModuleImportDecl) {
    if (ImportDecl *ImportD = dyn_cast_or_null<ImportDecl>(getCursorDecl(C)))
      return ImportD->getImportedModule();
  }

  return 0;
}

CXModule clang_Module_getParent(CXModule CXMod) {
  if (!CXMod)
    return 0;
  Module *Mod = static_cast<Module*>(CXMod);
  return Mod->Parent;
}

CXString clang_Module_getName(CXModule CXMod) {
  if (!CXMod)
    return createCXString("");
  Module *Mod = static_cast<Module*>(CXMod);
  return createCXString(Mod->Name);
}

CXString clang_Module_getFullName(CXModule CXMod) {
  if (!CXMod)
    return createCXString("");
  Module *Mod = static_cast<Module*>(CXMod);
  return createCXString(Mod->getFullModuleName());
}

unsigned clang_Module_getNumTopLevelHeaders(CXModule CXMod) {
  if (!CXMod)
    return 0;
  Module *Mod = static_cast<Module*>(CXMod);
  return Mod->TopHeaders.size();
}

CXFile clang_Module_getTopLevelHeader(CXModule CXMod, unsigned Index) {
  if (!CXMod)
    return 0;
  Module *Mod = static_cast<Module*>(CXMod);

  if (Index < Mod->TopHeaders.size())
    return const_cast<FileEntry *>(Mod->TopHeaders[Index]);

  return 0;
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

unsigned clang_CXXMethod_isVirtual(CXCursor C) {
  if (!clang_isDeclaration(C.kind))
    return 0;
  
  CXXMethodDecl *Method = 0;
  Decl *D = cxcursor::getCursorDecl(C);
  if (FunctionTemplateDecl *FunTmpl = dyn_cast_or_null<FunctionTemplateDecl>(D))
    Method = dyn_cast<CXXMethodDecl>(FunTmpl->getTemplatedDecl());
  else
    Method = dyn_cast_or_null<CXXMethodDecl>(D);
  return (Method && Method->isVirtual()) ? 1 : 0;
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
  
  return cxtype::MakeCXType(A->getInterface(), cxcursor::getCursorTU(C));  
}
} // end: extern "C"

//===----------------------------------------------------------------------===//
// Inspecting memory usage.
//===----------------------------------------------------------------------===//

typedef std::vector<CXTUResourceUsageEntry> MemUsageEntries;

static inline void createCXTUResourceUsageEntry(MemUsageEntries &entries,
                                              enum CXTUResourceUsageKind k,
                                              unsigned long amount) {
  CXTUResourceUsageEntry entry = { k, amount };
  entries.push_back(entry);
}

extern "C" {

const char *clang_getTUResourceUsageName(CXTUResourceUsageKind kind) {
  const char *str = "";
  switch (kind) {
    case CXTUResourceUsage_AST:
      str = "ASTContext: expressions, declarations, and types"; 
      break;
    case CXTUResourceUsage_Identifiers:
      str = "ASTContext: identifiers";
      break;
    case CXTUResourceUsage_Selectors:
      str = "ASTContext: selectors";
      break;
    case CXTUResourceUsage_GlobalCompletionResults:
      str = "Code completion: cached global results";
      break;
    case CXTUResourceUsage_SourceManagerContentCache:
      str = "SourceManager: content cache allocator";
      break;
    case CXTUResourceUsage_AST_SideTables:
      str = "ASTContext: side tables";
      break;
    case CXTUResourceUsage_SourceManager_Membuffer_Malloc:
      str = "SourceManager: malloc'ed memory buffers";
      break;
    case CXTUResourceUsage_SourceManager_Membuffer_MMap:
      str = "SourceManager: mmap'ed memory buffers";
      break;
    case CXTUResourceUsage_ExternalASTSource_Membuffer_Malloc:
      str = "ExternalASTSource: malloc'ed memory buffers";
      break;
    case CXTUResourceUsage_ExternalASTSource_Membuffer_MMap:
      str = "ExternalASTSource: mmap'ed memory buffers";
      break;
    case CXTUResourceUsage_Preprocessor:
      str = "Preprocessor: malloc'ed memory";
      break;
    case CXTUResourceUsage_PreprocessingRecord:
      str = "Preprocessor: PreprocessingRecord";
      break;
    case CXTUResourceUsage_SourceManager_DataStructures:
      str = "SourceManager: data structures and tables";
      break;
    case CXTUResourceUsage_Preprocessor_HeaderSearch:
      str = "Preprocessor: header search tables";
      break;
  }
  return str;
}

CXTUResourceUsage clang_getCXTUResourceUsage(CXTranslationUnit TU) {
  if (!TU) {
    CXTUResourceUsage usage = { (void*) 0, 0, 0 };
    return usage;
  }
  
  ASTUnit *astUnit = static_cast<ASTUnit*>(TU->TUData);
  OwningPtr<MemUsageEntries> entries(new MemUsageEntries());
  ASTContext &astContext = astUnit->getASTContext();
  
  // How much memory is used by AST nodes and types?
  createCXTUResourceUsageEntry(*entries, CXTUResourceUsage_AST,
    (unsigned long) astContext.getASTAllocatedMemory());

  // How much memory is used by identifiers?
  createCXTUResourceUsageEntry(*entries, CXTUResourceUsage_Identifiers,
    (unsigned long) astContext.Idents.getAllocator().getTotalMemory());

  // How much memory is used for selectors?
  createCXTUResourceUsageEntry(*entries, CXTUResourceUsage_Selectors,
    (unsigned long) astContext.Selectors.getTotalMemory());
  
  // How much memory is used by ASTContext's side tables?
  createCXTUResourceUsageEntry(*entries, CXTUResourceUsage_AST_SideTables,
    (unsigned long) astContext.getSideTableAllocatedMemory());
  
  // How much memory is used for caching global code completion results?
  unsigned long completionBytes = 0;
  if (GlobalCodeCompletionAllocator *completionAllocator =
      astUnit->getCachedCompletionAllocator().getPtr()) {
    completionBytes = completionAllocator->getTotalMemory();
  }
  createCXTUResourceUsageEntry(*entries,
                               CXTUResourceUsage_GlobalCompletionResults,
                               completionBytes);
  
  // How much memory is being used by SourceManager's content cache?
  createCXTUResourceUsageEntry(*entries,
          CXTUResourceUsage_SourceManagerContentCache,
          (unsigned long) astContext.getSourceManager().getContentCacheSize());
  
  // How much memory is being used by the MemoryBuffer's in SourceManager?
  const SourceManager::MemoryBufferSizes &srcBufs =
    astUnit->getSourceManager().getMemoryBufferSizes();
  
  createCXTUResourceUsageEntry(*entries,
                               CXTUResourceUsage_SourceManager_Membuffer_Malloc,
                               (unsigned long) srcBufs.malloc_bytes);
  createCXTUResourceUsageEntry(*entries,
                               CXTUResourceUsage_SourceManager_Membuffer_MMap,
                               (unsigned long) srcBufs.mmap_bytes);
  createCXTUResourceUsageEntry(*entries,
                               CXTUResourceUsage_SourceManager_DataStructures,
                               (unsigned long) astContext.getSourceManager()
                                .getDataStructureSizes());
  
  // How much memory is being used by the ExternalASTSource?
  if (ExternalASTSource *esrc = astContext.getExternalSource()) {
    const ExternalASTSource::MemoryBufferSizes &sizes =
      esrc->getMemoryBufferSizes();
    
    createCXTUResourceUsageEntry(*entries,
      CXTUResourceUsage_ExternalASTSource_Membuffer_Malloc,
                                 (unsigned long) sizes.malloc_bytes);
    createCXTUResourceUsageEntry(*entries,
      CXTUResourceUsage_ExternalASTSource_Membuffer_MMap,
                                 (unsigned long) sizes.mmap_bytes);
  }
  
  // How much memory is being used by the Preprocessor?
  Preprocessor &pp = astUnit->getPreprocessor();
  createCXTUResourceUsageEntry(*entries,
                               CXTUResourceUsage_Preprocessor,
                               pp.getTotalMemory());
  
  if (PreprocessingRecord *pRec = pp.getPreprocessingRecord()) {
    createCXTUResourceUsageEntry(*entries,
                                 CXTUResourceUsage_PreprocessingRecord,
                                 pRec->getTotalMemory());    
  }
  
  createCXTUResourceUsageEntry(*entries,
                               CXTUResourceUsage_Preprocessor_HeaderSearch,
                               pp.getHeaderSearchInfo().getTotalMemory());
  
  CXTUResourceUsage usage = { (void*) entries.get(),
                            (unsigned) entries->size(),
                            entries->size() ? &(*entries)[0] : 0 };
  entries.take();
  return usage;
}

void clang_disposeCXTUResourceUsage(CXTUResourceUsage usage) {
  if (usage.data)
    delete (MemUsageEntries*) usage.data;
}

} // end extern "C"

void clang::PrintLibclangResourceUsage(CXTranslationUnit TU) {
  CXTUResourceUsage Usage = clang_getCXTUResourceUsage(TU);
  for (unsigned I = 0; I != Usage.numEntries; ++I)
    fprintf(stderr, "  %s: %lu\n", 
            clang_getTUResourceUsageName(Usage.entries[I].kind),
            Usage.entries[I].amount);
  
  clang_disposeCXTUResourceUsage(Usage);
}

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

void clang::setThreadBackgroundPriority() {
  if (getenv("LIBCLANG_BGPRIO_DISABLE"))
    return;

  // FIXME: Move to llvm/Support and make it cross-platform.
#ifdef __APPLE__
  setpriority(PRIO_DARWIN_THREAD, 0, PRIO_DARWIN_BG);
#endif
}

void cxindex::printDiagsToStderr(ASTUnit *Unit) {
  if (!Unit)
    return;

  for (ASTUnit::stored_diag_iterator D = Unit->stored_diag_begin(), 
                                  DEnd = Unit->stored_diag_end();
       D != DEnd; ++D) {
    CXStoredDiagnostic Diag(*D, Unit->getASTContext().getLangOpts());
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

extern "C" {

CXString clang_getClangVersion() {
  return createCXString(getClangFullVersion());
}

} // end: extern "C"

