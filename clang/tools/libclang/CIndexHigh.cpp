//===- CIndexHigh.cpp - Higher level API functions ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CursorVisitor.h"
#include "CXCursor.h"
#include "CXSourceLocation.h"
#include "CXTranslationUnit.h"

#include "clang/Frontend/ASTUnit.h"
#include "clang/AST/DeclObjC.h"

using namespace clang;
using namespace cxcursor;

static void getTopOverriddenMethods(CXTranslationUnit TU,
                                    Decl *D,
                                    SmallVectorImpl<Decl *> &Methods) {
  if (!D)
    return;
  if (!isa<ObjCMethodDecl>(D) && !isa<CXXMethodDecl>(D))
    return;

  SmallVector<CXCursor, 8> Overridden;
  cxcursor::getOverriddenCursors(cxcursor::MakeCXCursor(D, TU), Overridden);
  
  if (Overridden.empty()) {
    Methods.push_back(D->getCanonicalDecl());
    return;
  }

  for (SmallVector<CXCursor, 8>::iterator
         I = Overridden.begin(), E = Overridden.end(); I != E; ++I)
    getTopOverriddenMethods(TU, cxcursor::getCursorDecl(*I), Methods);
}

namespace {

struct FindFileIdRefVisitData {
  CXTranslationUnit TU;
  FileID FID;
  Decl *Dcl;
  int SelectorIdIdx;
  CXCursorAndRangeVisitor visitor;

  typedef SmallVector<Decl *, 8> TopMethodsTy;
  TopMethodsTy TopMethods;

  FindFileIdRefVisitData(CXTranslationUnit TU, FileID FID,
                         Decl *D, int selectorIdIdx,
                         CXCursorAndRangeVisitor visitor)
    : TU(TU), FID(FID), SelectorIdIdx(selectorIdIdx), visitor(visitor) {
    Dcl = getCanonical(D);
    getTopOverriddenMethods(TU, Dcl, TopMethods);
  }

  ASTContext &getASTContext() const {
    return static_cast<ASTUnit *>(TU->TUData)->getASTContext();
  }

  /// \brief We are looking to find all semantically relevant identifiers,
  /// so the definition of "canonical" here is different than in the AST, e.g.
  ///
  /// \code
  ///   class C {
  ///     C() {}
  ///   };
  /// \endcode
  ///
  /// we consider the canonical decl of the constructor decl to be the class
  /// itself, so both 'C' can be highlighted.
  Decl *getCanonical(Decl *D) const {
    if (!D)
      return 0;

    D = D->getCanonicalDecl();

    if (ObjCImplDecl *ImplD = dyn_cast<ObjCImplDecl>(D)) {
      if (ImplD->getClassInterface())
        return getCanonical(ImplD->getClassInterface());

    } else if (CXXConstructorDecl *CXXCtorD = dyn_cast<CXXConstructorDecl>(D)) {
      return getCanonical(CXXCtorD->getParent());
    }
    
    return D;
  }

  bool isHit(Decl *D) const {
    if (!D)
      return false;

    D = getCanonical(D);
    if (D == Dcl)
      return true;

    if (isa<ObjCMethodDecl>(D) || isa<CXXMethodDecl>(D))
      return isOverriddingMethod(D);

    return false;
  }

private:
  bool isOverriddingMethod(Decl *D) const {
    if (std::find(TopMethods.begin(), TopMethods.end(), D) !=
          TopMethods.end())
      return true;

    TopMethodsTy methods;
    getTopOverriddenMethods(TU, D, methods);
    for (TopMethodsTy::iterator
           I = methods.begin(), E = methods.end(); I != E; ++I) {
      if (std::find(TopMethods.begin(), TopMethods.end(), *I) !=
            TopMethods.end())
        return true;
    }

    return false;
  }
};

} // end anonymous namespace.

/// \brief For a macro \arg Loc, returns the file spelling location and sets
/// to \arg isMacroArg whether the spelling resides inside a macro definition or
/// a macro argument.
static SourceLocation getFileSpellingLoc(SourceManager &SM,
                                         SourceLocation Loc,
                                         bool &isMacroArg) {
  assert(Loc.isMacroID());
  SourceLocation SpellLoc = SM.getImmediateSpellingLoc(Loc);
  if (SpellLoc.isMacroID())
    return getFileSpellingLoc(SM, SpellLoc, isMacroArg);
  
  isMacroArg = SM.isMacroArgExpansion(Loc);
  return SpellLoc;
}

static enum CXChildVisitResult findFileIdRefVisit(CXCursor cursor,
                                                  CXCursor parent,
                                                  CXClientData client_data) {
  CXCursor declCursor = clang_getCursorReferenced(cursor);
  if (!clang_isDeclaration(declCursor.kind))
    return CXChildVisit_Recurse;

  Decl *D = cxcursor::getCursorDecl(declCursor);
  if (!D)
    return CXChildVisit_Continue;

  FindFileIdRefVisitData *data = (FindFileIdRefVisitData *)client_data;
  if (data->isHit(D)) {
    cursor = cxcursor::getSelectorIdentifierCursor(data->SelectorIdIdx, cursor);

    // We are looking for identifiers to highlight so for objc methods (and
    // not a parameter) we can only highlight the selector identifiers.
    if ((cursor.kind == CXCursor_ObjCClassMethodDecl ||
         cursor.kind == CXCursor_ObjCInstanceMethodDecl) &&
         cxcursor::getSelectorIdentifierIndex(cursor) == -1)
      return CXChildVisit_Recurse;

    if (clang_isExpression(cursor.kind)) {
      if (cursor.kind == CXCursor_DeclRefExpr ||
          cursor.kind == CXCursor_MemberRefExpr) {
        // continue..

      } else if (cursor.kind == CXCursor_ObjCMessageExpr &&
                 cxcursor::getSelectorIdentifierIndex(cursor) != -1) {
        // continue..
                
      } else
        return CXChildVisit_Recurse;
    }

    SourceLocation
      Loc = cxloc::translateSourceLocation(clang_getCursorLocation(cursor));
    SourceLocation SelIdLoc = cxcursor::getSelectorIdentifierLoc(cursor);
    if (SelIdLoc.isValid())
      Loc = SelIdLoc;

    ASTContext &Ctx = data->getASTContext();
    SourceManager &SM = Ctx.getSourceManager();
    bool isInMacroDef = false;
    if (Loc.isMacroID()) {
      bool isMacroArg;
      Loc = getFileSpellingLoc(SM, Loc, isMacroArg);
      isInMacroDef = !isMacroArg;
    }

    // We are looking for identifiers in a specific file.
    std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(Loc);
    if (LocInfo.first != data->FID)
      return CXChildVisit_Recurse;

    if (isInMacroDef) {
      // FIXME: For a macro definition make sure that all expansions
      // of it expand to the same reference before allowing to point to it.
      return CXChildVisit_Recurse;
    }

    data->visitor.visit(data->visitor.context, cursor,
                        cxloc::translateSourceRange(Ctx, Loc));
  }
  return CXChildVisit_Recurse;
}

static void findIdRefsInFile(CXTranslationUnit TU, CXCursor declCursor,
                           const FileEntry *File,
                           CXCursorAndRangeVisitor Visitor) {
  assert(clang_isDeclaration(declCursor.kind));
  ASTUnit *Unit = static_cast<ASTUnit*>(TU->TUData);
  SourceManager &SM = Unit->getSourceManager();

  FileID FID = SM.translateFile(File);
  Decl *Dcl = cxcursor::getCursorDecl(declCursor);
  if (!Dcl)
    return;

  FindFileIdRefVisitData data(TU, FID, Dcl,
                              cxcursor::getSelectorIdentifierIndex(declCursor),
                              Visitor);

  if (DeclContext *DC = Dcl->getParentFunctionOrMethod()) {
    clang_visitChildren(cxcursor::MakeCXCursor(cast<Decl>(DC), TU),
                        findFileIdRefVisit, &data);
    return;
  }

  SourceRange Range(SM.getLocForStartOfFile(FID), SM.getLocForEndOfFile(FID));
  CursorVisitor FindIdRefsVisitor(TU,
                                  findFileIdRefVisit, &data,
                                  /*VisitPreprocessorLast=*/true,
                                  /*VisitIncludedEntities=*/false,
                                  Range,
                                  /*VisitDeclsOnly=*/true);
  FindIdRefsVisitor.visitFileRegion();
}

namespace {

struct FindFileMacroRefVisitData {
  ASTUnit &Unit;
  const FileEntry *File;
  const IdentifierInfo *Macro;
  CXCursorAndRangeVisitor visitor;

  FindFileMacroRefVisitData(ASTUnit &Unit, const FileEntry *File,
                            const IdentifierInfo *Macro,
                            CXCursorAndRangeVisitor visitor)
    : Unit(Unit), File(File), Macro(Macro), visitor(visitor) { }

  ASTContext &getASTContext() const {
    return Unit.getASTContext();
  }
};

} // anonymous namespace

static enum CXChildVisitResult findFileMacroRefVisit(CXCursor cursor,
                                                     CXCursor parent,
                                                     CXClientData client_data) {
  const IdentifierInfo *Macro = 0;
  if (cursor.kind == CXCursor_MacroDefinition)
    Macro = getCursorMacroDefinition(cursor)->getName();
  else if (cursor.kind == CXCursor_MacroExpansion)
    Macro = getCursorMacroExpansion(cursor)->getName();
  if (!Macro)
    return CXChildVisit_Continue;

  FindFileMacroRefVisitData *data = (FindFileMacroRefVisitData *)client_data;
  if (data->Macro != Macro)
    return CXChildVisit_Continue;

  SourceLocation
    Loc = cxloc::translateSourceLocation(clang_getCursorLocation(cursor));

  ASTContext &Ctx = data->getASTContext();
  SourceManager &SM = Ctx.getSourceManager();
  bool isInMacroDef = false;
  if (Loc.isMacroID()) {
    bool isMacroArg;
    Loc = getFileSpellingLoc(SM, Loc, isMacroArg);
    isInMacroDef = !isMacroArg;
  }

  // We are looking for identifiers in a specific file.
  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(Loc);
  if (SM.getFileEntryForID(LocInfo.first) != data->File)
    return CXChildVisit_Continue;

  if (isInMacroDef) {
    // FIXME: For a macro definition make sure that all expansions
    // of it expand to the same reference before allowing to point to it.
    return CXChildVisit_Continue;
  }

  data->visitor.visit(data->visitor.context, cursor,
                      cxloc::translateSourceRange(Ctx, Loc));
  return CXChildVisit_Continue;
}

static void findMacroRefsInFile(CXTranslationUnit TU, CXCursor Cursor,
                                const FileEntry *File,
                                CXCursorAndRangeVisitor Visitor) {
  if (Cursor.kind != CXCursor_MacroDefinition &&
      Cursor.kind != CXCursor_MacroExpansion)
    return;

  ASTUnit *Unit = static_cast<ASTUnit*>(TU->TUData);
  SourceManager &SM = Unit->getSourceManager();

  FileID FID = SM.translateFile(File);
  const IdentifierInfo *Macro = 0;
  if (Cursor.kind == CXCursor_MacroDefinition)
    Macro = getCursorMacroDefinition(Cursor)->getName();
  else
    Macro = getCursorMacroExpansion(Cursor)->getName();
  if (!Macro)
    return;

  FindFileMacroRefVisitData data(*Unit, File, Macro, Visitor);

  SourceRange Range(SM.getLocForStartOfFile(FID), SM.getLocForEndOfFile(FID));
  CursorVisitor FindMacroRefsVisitor(TU,
                                  findFileMacroRefVisit, &data,
                                  /*VisitPreprocessorLast=*/false,
                                  /*VisitIncludedEntities=*/false,
                                  Range);
  FindMacroRefsVisitor.visitPreprocessedEntitiesInRegion();
}


//===----------------------------------------------------------------------===//
// libclang public APIs.
//===----------------------------------------------------------------------===//

extern "C" {

void clang_findReferencesInFile(CXCursor cursor, CXFile file,
                                CXCursorAndRangeVisitor visitor) {
  bool Logging = ::getenv("LIBCLANG_LOGGING");

  if (clang_Cursor_isNull(cursor)) {
    if (Logging)
      llvm::errs() << "clang_findReferencesInFile: Null cursor\n";
    return;
  }
  if (cursor.kind == CXCursor_NoDeclFound) {
    if (Logging)
      llvm::errs() << "clang_findReferencesInFile: Got CXCursor_NoDeclFound\n";
    return;
  }
  if (!file) {
    if (Logging)
      llvm::errs() << "clang_findReferencesInFile: Null file\n";
    return;
  }
  if (!visitor.visit) {
    if (Logging)
      llvm::errs() << "clang_findReferencesInFile: Null visitor\n";
    return;
  }

  ASTUnit *CXXUnit = cxcursor::getCursorASTUnit(cursor);
  if (!CXXUnit)
    return;

  ASTUnit::ConcurrencyCheck Check(*CXXUnit);

  if (cursor.kind == CXCursor_MacroDefinition ||
      cursor.kind == CXCursor_MacroExpansion) {
    findMacroRefsInFile(cxcursor::getCursorTU(cursor),
                        cursor,
                        static_cast<const FileEntry *>(file),
                        visitor);
    return;
  }

  // We are interested in semantics of identifiers so for C++ constructor exprs
  // prefer type references, e.g.:
  //
  //  return MyStruct();
  //
  // for 'MyStruct' we'll have a cursor pointing at the constructor decl but
  // we are actually interested in the type declaration.
  cursor = cxcursor::getTypeRefCursor(cursor);

  CXCursor refCursor = clang_getCursorReferenced(cursor);

  if (!clang_isDeclaration(refCursor.kind)) {
    if (Logging)
      llvm::errs() << "clang_findReferencesInFile: cursor is not referencing a "
                      "declaration\n";
    return;
  }

  findIdRefsInFile(cxcursor::getCursorTU(cursor),
                   refCursor,
                   static_cast<const FileEntry *>(file),
                   visitor);
}

static enum CXVisitorResult _visitCursorAndRange(void *context,
                                                 CXCursor cursor,
                                                 CXSourceRange range) {
  CXCursorAndRangeVisitorBlock block = (CXCursorAndRangeVisitorBlock)context;
  return INVOKE_BLOCK2(block, cursor, range);
}

void clang_findReferencesInFileWithBlock(CXCursor cursor,
                                         CXFile file,
                                         CXCursorAndRangeVisitorBlock block) {
  CXCursorAndRangeVisitor visitor = { block,
                                      block ? _visitCursorAndRange : 0 };
  return clang_findReferencesInFile(cursor, file, visitor);
}

} // end: extern "C"

