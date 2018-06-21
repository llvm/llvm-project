//===- CXCursor.h - Routines for manipulating CXCursors -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines routines for manipulating CXCursors.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_LIBCLANG_CXCURSOR_H
#define LLVM_CLANG_TOOLS_LIBCLANG_CXCURSOR_H

#include "clang-c/Index.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/PointerUnion.h"
#include <utility>

namespace clang {

class ASTContext;
class ASTUnit;
class Attr;
class CXXBaseSpecifier;
class Decl;
class Expr;
class FieldDecl;
class InclusionDirective;
class LabelStmt;
class MacroDefinitionRecord;
class MacroExpansion;
class NamedDecl;
class ObjCInterfaceDecl;
class ObjCProtocolDecl;
class OverloadedTemplateStorage;
class OverloadExpr;
class Stmt;
class TemplateDecl;
class TemplateName;
class TypeDecl;
class VarDecl;
class IdentifierInfo;
  
namespace cxcursor {

CXCursor getCursor(CXTranslationUnit, SourceLocation);
  
CXCursor MakeCXCursor(const clang::Attr *A, const clang::Decl *Parent,
                      CXTranslationUnit TU);
CXCursor MakeCXCursor(const clang::Decl *D, CXTranslationUnit TU,
                      SourceRange RegionOfInterest = SourceRange(),
                      bool FirstInDeclGroup = true);
CXCursor MakeCXCursor(const clang::Stmt *S, const clang::Decl *Parent,
                      CXTranslationUnit TU,
                      SourceRange RegionOfInterest = SourceRange());
CXCursor MakeCXCursorInvalid(CXCursorKind K, CXTranslationUnit TU = nullptr);

/// \brief Create an Objective-C superclass reference at the given location.
CXCursor MakeCursorObjCSuperClassRef(ObjCInterfaceDecl *Super, 
                                     SourceLocation Loc, 
                                     CXTranslationUnit TU);

/// \brief Unpack an ObjCSuperClassRef cursor into the interface it references
/// and optionally the location where the reference occurred.
std::pair<const ObjCInterfaceDecl *, SourceLocation>
  getCursorObjCSuperClassRef(CXCursor C);

/// \brief Create an Objective-C protocol reference at the given location.
CXCursor MakeCursorObjCProtocolRef(const ObjCProtocolDecl *Proto,
                                   SourceLocation Loc, 
                                   CXTranslationUnit TU);

/// \brief Unpack an ObjCProtocolRef cursor into the protocol it references
/// and optionally the location where the reference occurred.
std::pair<const ObjCProtocolDecl *, SourceLocation>
  getCursorObjCProtocolRef(CXCursor C);

/// \brief Create an Objective-C class reference at the given location.
CXCursor MakeCursorObjCClassRef(const ObjCInterfaceDecl *Class,
                                SourceLocation Loc, 
                                CXTranslationUnit TU);

/// \brief Unpack an ObjCClassRef cursor into the class it references
/// and optionally the location where the reference occurred.
std::pair<const ObjCInterfaceDecl *, SourceLocation>
  getCursorObjCClassRef(CXCursor C);

/// \brief Create a type reference at the given location.
CXCursor MakeCursorTypeRef(const TypeDecl *Type, SourceLocation Loc,
                           CXTranslationUnit TU);
                               
/// \brief Unpack a TypeRef cursor into the class it references
/// and optionally the location where the reference occurred.
std::pair<const TypeDecl *, SourceLocation> getCursorTypeRef(CXCursor C);

/// \brief Create a reference to a template at the given location.
CXCursor MakeCursorTemplateRef(const TemplateDecl *Template, SourceLocation Loc,
                               CXTranslationUnit TU);

/// \brief Unpack a TemplateRef cursor into the template it references and
/// the location where the reference occurred.
std::pair<const TemplateDecl *, SourceLocation>
  getCursorTemplateRef(CXCursor C);

/// \brief Create a reference to a namespace or namespace alias at the given 
/// location.
CXCursor MakeCursorNamespaceRef(const NamedDecl *NS, SourceLocation Loc,
                                CXTranslationUnit TU);

/// \brief Unpack a NamespaceRef cursor into the namespace or namespace alias
/// it references and the location where the reference occurred.
std::pair<const NamedDecl *, SourceLocation> getCursorNamespaceRef(CXCursor C);

/// \brief Create a reference to a variable at the given location.
CXCursor MakeCursorVariableRef(const VarDecl *Var, SourceLocation Loc, 
                               CXTranslationUnit TU);

/// \brief Unpack a VariableRef cursor into the variable it references and the
/// location where the where the reference occurred.
std::pair<const VarDecl *, SourceLocation> getCursorVariableRef(CXCursor C);

/// \brief Create a reference to a field at the given location.
CXCursor MakeCursorMemberRef(const FieldDecl *Field, SourceLocation Loc, 
                             CXTranslationUnit TU);
  
/// \brief Unpack a MemberRef cursor into the field it references and the 
/// location where the reference occurred.
std::pair<const FieldDecl *, SourceLocation> getCursorMemberRef(CXCursor C);

/// \brief Create a CXX base specifier cursor.
CXCursor MakeCursorCXXBaseSpecifier(const CXXBaseSpecifier *B,
                                    CXTranslationUnit TU);

/// \brief Unpack a CXXBaseSpecifier cursor into a CXXBaseSpecifier.
const CXXBaseSpecifier *getCursorCXXBaseSpecifier(CXCursor C);

/// \brief Create a preprocessing directive cursor.
CXCursor MakePreprocessingDirectiveCursor(SourceRange Range,
                                          CXTranslationUnit TU);

/// \brief Unpack a given preprocessing directive to retrieve its source range.
SourceRange getCursorPreprocessingDirective(CXCursor C);

/// \brief Create a macro definition cursor.
CXCursor MakeMacroDefinitionCursor(const MacroDefinitionRecord *,
                                   CXTranslationUnit TU);

/// \brief Unpack a given macro definition cursor to retrieve its
/// source range.
const MacroDefinitionRecord *getCursorMacroDefinition(CXCursor C);

/// \brief Create a macro expansion cursor.
CXCursor MakeMacroExpansionCursor(MacroExpansion *, CXTranslationUnit TU);

/// \brief Create a "pseudo" macro expansion cursor, using a macro definition
/// and a source location.
CXCursor MakeMacroExpansionCursor(MacroDefinitionRecord *, SourceLocation Loc,
                                  CXTranslationUnit TU);

/// \brief Wraps a macro expansion cursor and provides a common interface
/// for a normal macro expansion cursor or a "pseudo" one.
///
/// "Pseudo" macro expansion cursors (essentially a macro definition along with
/// a source location) are created in special cases, for example they can be
/// created for identifiers inside macro definitions, if these identifiers are
/// macro names.
class MacroExpansionCursor {
  CXCursor C;

  bool isPseudo() const { return C.data[1] != nullptr; }
  const MacroDefinitionRecord *getAsMacroDefinition() const {
    assert(isPseudo());
    return static_cast<const MacroDefinitionRecord *>(C.data[0]);
  }
  const MacroExpansion *getAsMacroExpansion() const {
    assert(!isPseudo());
    return static_cast<const MacroExpansion *>(C.data[0]);
  }
  SourceLocation getPseudoLoc() const {
    assert(isPseudo());
    return SourceLocation::getFromPtrEncoding(C.data[1]);
  }

public:
  MacroExpansionCursor(CXCursor C) : C(C) {
    assert(C.kind == CXCursor_MacroExpansion);
  }

  const IdentifierInfo *getName() const;
  const MacroDefinitionRecord *getDefinition() const;
  SourceRange getSourceRange() const;
};

/// \brief Unpack a given macro expansion cursor to retrieve its info.
static inline MacroExpansionCursor getCursorMacroExpansion(CXCursor C) {
  return C;
}

/// \brief Create an inclusion directive cursor.
CXCursor MakeInclusionDirectiveCursor(InclusionDirective *,
                                      CXTranslationUnit TU);

/// \brief Unpack a given inclusion directive cursor to retrieve its
/// source range.
const InclusionDirective *getCursorInclusionDirective(CXCursor C);

/// \brief Create a label reference at the given location.
CXCursor MakeCursorLabelRef(LabelStmt *Label, SourceLocation Loc,
                            CXTranslationUnit TU);

/// \brief Unpack a label reference into the label statement it refers to and
/// the location of the reference.
std::pair<const LabelStmt *, SourceLocation> getCursorLabelRef(CXCursor C);

/// \brief Create a overloaded declaration reference cursor for an expression.
CXCursor MakeCursorOverloadedDeclRef(const OverloadExpr *E,
                                     CXTranslationUnit TU);

/// \brief Create a overloaded declaration reference cursor for a declaration.
CXCursor MakeCursorOverloadedDeclRef(const Decl *D, SourceLocation Location,
                                     CXTranslationUnit TU);

/// \brief Create a overloaded declaration reference cursor for a template name.
CXCursor MakeCursorOverloadedDeclRef(TemplateName Template, 
                                     SourceLocation Location,
                                     CXTranslationUnit TU);

/// \brief Internal storage for an overloaded declaration reference cursor;
typedef llvm::PointerUnion3<const OverloadExpr *, const Decl *,
                            OverloadedTemplateStorage *>
  OverloadedDeclRefStorage;
  
/// \brief Unpack an overloaded declaration reference into an expression,
/// declaration, or template name along with the source location.
std::pair<OverloadedDeclRefStorage, SourceLocation>
  getCursorOverloadedDeclRef(CXCursor C);
  
const Decl *getCursorDecl(CXCursor Cursor);
const Expr *getCursorExpr(CXCursor Cursor);
const Stmt *getCursorStmt(CXCursor Cursor);
const Attr *getCursorAttr(CXCursor Cursor);
const Decl *getCursorParentDecl(CXCursor Cursor);

ASTContext &getCursorContext(CXCursor Cursor);
ASTUnit *getCursorASTUnit(CXCursor Cursor);
CXTranslationUnit getCursorTU(CXCursor Cursor);

void getOverriddenCursors(CXCursor cursor,
                          SmallVectorImpl<CXCursor> &overridden);
  
/// \brief Create an opaque  pool used for fast generation of overriden
/// CXCursor arrays.
void *createOverridenCXCursorsPool();

/// \brief Dispose of the overriden CXCursors pool.
void disposeOverridenCXCursorsPool(void *pool);
  
/// \brief Returns a index/location pair for a selector identifier if the cursor
/// points to one.
std::pair<int, SourceLocation> getSelectorIdentifierIndexAndLoc(CXCursor);
static inline int getSelectorIdentifierIndex(CXCursor cursor) {
  return getSelectorIdentifierIndexAndLoc(cursor).first;
}
static inline SourceLocation getSelectorIdentifierLoc(CXCursor cursor) {
  return getSelectorIdentifierIndexAndLoc(cursor).second;
}

CXCursor getSelectorIdentifierCursor(int SelIdx, CXCursor cursor);

static inline CXCursor getTypeRefedCallExprCursor(CXCursor cursor) {
  CXCursor newCursor = cursor;
  if (cursor.kind == CXCursor_CallExpr)
    newCursor.xdata = 1;
  return newCursor;
}

CXCursor getTypeRefCursor(CXCursor cursor);

/// \brief Generate a USR for \arg D and put it in \arg Buf.
/// \returns true if no USR was computed or the result should be ignored,
/// false otherwise.
bool getDeclCursorUSR(const Decl *D, SmallVectorImpl<char> &Buf);

bool operator==(CXCursor X, CXCursor Y);
  
inline bool operator!=(CXCursor X, CXCursor Y) {
  return !(X == Y);
}

/// \brief Return true if the cursor represents a declaration that is the
/// first in a declaration group.
bool isFirstInDeclGroup(CXCursor C);

}} // end namespace: clang::cxcursor

#endif
