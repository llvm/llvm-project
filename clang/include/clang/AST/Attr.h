//===--- Attr.h - Classes for representing attributes ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Attr interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_ATTR_H
#define LLVM_CLANG_AST_ATTR_H

#include "clang/AST/AttrIterator.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Basic/AttrKinds.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/Sanitizers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/VersionTuple.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>

namespace clang {
  class ASTContext;
  class IdentifierInfo;
  class ObjCInterfaceDecl;
  class Expr;
  class QualType;
  class FunctionDecl;
  class TypeSourceInfo;

/// Attr - This represents one attribute.
class Attr {
private:
  SourceRange Range;
  unsigned AttrKind : 16;

protected:
  /// An index into the spelling list of an
  /// attribute defined in Attr.td file.
  unsigned SpellingListIndex : 4;
  bool Inherited : 1;
  bool IsPackExpansion : 1;
  bool Implicit : 1;
  bool IsLateParsed : 1;
  bool DuplicatesAllowed : 1;

  void *operator new(size_t bytes) LLVM_NOEXCEPT {
    llvm_unreachable("Attrs cannot be allocated with regular 'new'.");
  }
  void operator delete(void *data) LLVM_NOEXCEPT {
    llvm_unreachable("Attrs cannot be released with regular 'delete'.");
  }

public:
  // Forward so that the regular new and delete do not hide global ones.
  void *operator new(size_t Bytes, ASTContext &C,
                     size_t Alignment = 8) LLVM_NOEXCEPT {
    return ::operator new(Bytes, C, Alignment);
  }
  void operator delete(void *Ptr, ASTContext &C,
                       size_t Alignment) LLVM_NOEXCEPT {
    return ::operator delete(Ptr, C, Alignment);
  }

protected:
  Attr(attr::Kind AK, SourceRange R, unsigned SpellingListIndex,
       bool IsLateParsed, bool DuplicatesAllowed)
    : Range(R), AttrKind(AK), SpellingListIndex(SpellingListIndex),
      Inherited(false), IsPackExpansion(false), Implicit(false),
      IsLateParsed(IsLateParsed), DuplicatesAllowed(DuplicatesAllowed) {}

public:

  attr::Kind getKind() const {
    return static_cast<attr::Kind>(AttrKind);
  }
  
  unsigned getSpellingListIndex() const { return SpellingListIndex; }
  const char *getSpelling() const;

  SourceLocation getLocation() const { return Range.getBegin(); }
  SourceRange getRange() const { return Range; }
  void setRange(SourceRange R) { Range = R; }

  bool isInherited() const { return Inherited; }

  /// \brief Returns true if the attribute has been implicitly created instead
  /// of explicitly written by the user.
  bool isImplicit() const { return Implicit; }
  void setImplicit(bool I) { Implicit = I; }

  void setPackExpansion(bool PE) { IsPackExpansion = PE; }
  bool isPackExpansion() const { return IsPackExpansion; }

  // Clone this attribute.
  Attr *clone(ASTContext &C) const;

  bool isLateParsed() const { return IsLateParsed; }

  // Pretty print this attribute.
  void printPretty(raw_ostream &OS, const PrintingPolicy &Policy) const;

  /// \brief By default, attributes cannot be duplicated when being merged;
  /// however, an attribute can override this. Returns true if the attribute
  /// can be duplicated when merging.
  bool duplicatesAllowed() const { return DuplicatesAllowed; }
};

class InheritableAttr : public Attr {
protected:
  InheritableAttr(attr::Kind AK, SourceRange R, unsigned SpellingListIndex,
                  bool IsLateParsed, bool DuplicatesAllowed)
      : Attr(AK, R, SpellingListIndex, IsLateParsed, DuplicatesAllowed) {}

public:
  void setInherited(bool I) { Inherited = I; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    return A->getKind() <= attr::LAST_INHERITABLE;
  }
};

class InheritableParamAttr : public InheritableAttr {
protected:
  InheritableParamAttr(attr::Kind AK, SourceRange R, unsigned SpellingListIndex,
                       bool IsLateParsed, bool DuplicatesAllowed)
      : InheritableAttr(AK, R, SpellingListIndex, IsLateParsed,
                        DuplicatesAllowed) {}

public:
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    // Relies on relative order of enum emission with respect to MS inheritance
    // attrs.
    return A->getKind() <= attr::LAST_INHERITABLE_PARAM;
  }
};

#include "clang/AST/Attrs.inc"

inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           const Attr *At) {
  DB.AddTaggedVal(reinterpret_cast<intptr_t>(At),
                  DiagnosticsEngine::ak_attr);
  return DB;
}

inline const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
                                           const Attr *At) {
  PD.AddTaggedVal(reinterpret_cast<intptr_t>(At),
                  DiagnosticsEngine::ak_attr);
  return PD;
}
}  // end namespace clang

#endif
