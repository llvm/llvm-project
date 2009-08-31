//===--- TypeLoc.h - Type Source Info Wrapper -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TypeLoc interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_TYPELOC_H
#define LLVM_CLANG_AST_TYPELOC_H

#include "clang/AST/Type.h"
#include "clang/AST/TypeVisitor.h"

namespace clang {
  class ParmVarDecl;
  class TypeSpecLoc;
  class DeclaratorInfo;

/// \brief Base wrapper for a particular "section" of type source info.
///
/// A client should use the TypeLoc subclasses through cast/dyn_cast in order to
/// get at the actual information.
class TypeLoc {
protected:
  QualType Ty;
  void *Data;
  
  TypeLoc(QualType ty, void *data) : Ty(ty), Data(data) { }
  static TypeLoc Create(QualType ty, void *data) { return TypeLoc(ty,data); }
  friend class DeclaratorInfo;
public:
  TypeLoc() : Data(0) { }

  bool isNull() const { return Ty.isNull(); }
  operator bool() const { return !isNull(); }

  /// \brief Returns the size of type source info data block for the given type.
  static unsigned getFullDataSizeForType(QualType Ty);

  /// \brief Get the type for which this source info wrapper provides
  /// information.
  QualType getSourceType() const { return Ty; }

  /// \brief Find the TypeSpecLoc that is part of this TypeLoc.
  TypeSpecLoc getTypeSpecLoc() const;

  /// \brief Find the TypeSpecLoc that is part of this TypeLoc and return its
  /// SourceRange.
  SourceRange getTypeSpecRange() const;

  /// \brief Returns the size of the type source info data block.
  unsigned getFullDataSize() const;

  /// \brief Get the next TypeLoc pointed by this TypeLoc, e.g for "int*" the
  /// TypeLoc is a PointerLoc and next TypeLoc is for "int".
  TypeLoc getNextTypeLoc() const;

  friend bool operator==(const TypeLoc &LHS, const TypeLoc &RHS) {
    return LHS.Ty == RHS.Ty && LHS.Data == RHS.Data;
  }

  friend bool operator!=(const TypeLoc &LHS, const TypeLoc &RHS) {
    return !(LHS == RHS);
  }

  static bool classof(const TypeLoc *TL) { return true; }
};

/// \brief Base wrapper of type source info data for type-spec types.
class TypeSpecLoc : public TypeLoc  {
public:
  SourceRange getSourceRange() const;

  static bool classof(const TypeLoc *TL);
  static bool classof(const TypeSpecLoc *TL) { return true; }
};

/// \brief Base wrapper of type source info data for types part of a declarator,
/// excluding type-spec types.
class DeclaratorLoc : public TypeLoc  {
public:
  /// \brief Find the TypeSpecLoc that is part of this DeclaratorLoc.
  TypeSpecLoc getTypeSpecLoc() const;  

  static bool classof(const TypeLoc *TL);
  static bool classof(const DeclaratorLoc *TL) { return true; }
};

/// \brief The default wrapper for type-spec types that are not handled by
/// another specific wrapper.
class DefaultTypeSpecLoc : public TypeSpecLoc {
  struct Info {
    SourceLocation StartLoc;
  };
  
public:
  SourceLocation getStartLoc() const {
    return static_cast<Info*>(Data)->StartLoc;
  }
  void setStartLoc(SourceLocation Loc) {
    static_cast<Info*>(Data)->StartLoc = Loc;
  }
  SourceRange getSourceRange() const {
    return SourceRange(getStartLoc(), getStartLoc());
  }

  /// \brief Returns the size of the type source info data block that is
  /// specific to this type.
  unsigned getLocalDataSize() const { return sizeof(Info); }

  /// \brief Returns the size of the type source info data block.
  unsigned getFullDataSize() const { return getLocalDataSize(); }

  static bool classof(const TypeLoc *TL);
  static bool classof(const DefaultTypeSpecLoc *TL) { return true; }
};

/// \brief Wrapper for source info for typedefs.
class TypedefLoc : public TypeSpecLoc {
  struct Info {
    SourceLocation NameLoc;
  };

public:
  SourceLocation getNameLoc() const {
    return static_cast<Info*>(Data)->NameLoc;
  }
  void setNameLoc(SourceLocation Loc) {
    static_cast<Info*>(Data)->NameLoc = Loc;
  }
  SourceRange getSourceRange() const {
    return SourceRange(getNameLoc(), getNameLoc());
  }
  
  /// \brief Returns the size of the type source info data block that is
  /// specific to this type.
  unsigned getLocalDataSize() const { return sizeof(Info); }

  /// \brief Returns the size of the type source info data block.
  unsigned getFullDataSize() const { return getLocalDataSize(); }

  static bool classof(const TypeLoc *TL);
  static bool classof(const TypedefLoc *TL) { return true; }
};

/// \brief Wrapper for source info for pointers.
class PointerLoc : public DeclaratorLoc {
  struct Info {
    SourceLocation StarLoc;
  };

public:
  SourceLocation getStarLoc() const {
    return static_cast<Info*>(Data)->StarLoc;
  }
  void setStarLoc(SourceLocation Loc) {
    static_cast<Info*>(Data)->StarLoc = Loc;
  }

  TypeLoc getPointeeLoc() const {
    void *Next = static_cast<char*>(Data) + getLocalDataSize();
    return Create(cast<PointerType>(Ty)->getPointeeType(), Next);
  }

  /// \brief Find the TypeSpecLoc that is part of this PointerLoc.
  TypeSpecLoc getTypeSpecLoc() const {
    return getPointeeLoc().getTypeSpecLoc();
  }

  SourceRange getSourceRange() const {
    return SourceRange(getStarLoc(), getStarLoc());
  }

  /// \brief Returns the size of the type source info data block that is
  /// specific to this type.
  unsigned getLocalDataSize() const { return sizeof(Info); }

  /// \brief Returns the size of the type source info data block.
  unsigned getFullDataSize() const {
    return getLocalDataSize() + getPointeeLoc().getFullDataSize();
  }

  static bool classof(const TypeLoc *TL);
  static bool classof(const PointerLoc *TL) { return true; }
};

/// \brief Wrapper for source info for block pointers.
class BlockPointerLoc : public DeclaratorLoc {
  struct Info {
    SourceLocation CaretLoc;
  };

public:
  SourceLocation getCaretLoc() const {
    return static_cast<Info*>(Data)->CaretLoc;
  }
  void setCaretLoc(SourceLocation Loc) {
    static_cast<Info*>(Data)->CaretLoc = Loc;
  }

  TypeLoc getPointeeLoc() const {
    void *Next = static_cast<char*>(Data) + getLocalDataSize();
    return Create(cast<BlockPointerType>(Ty)->getPointeeType(), Next);
  }

  /// \brief Find the TypeSpecLoc that is part of this BlockPointerLoc.
  TypeSpecLoc getTypeSpecLoc() const {
    return getPointeeLoc().getTypeSpecLoc();
  }

  SourceRange getSourceRange() const {
    return SourceRange(getCaretLoc(), getCaretLoc());
  }

  /// \brief Returns the size of the type source info data block that is
  /// specific to this type.
  unsigned getLocalDataSize() const { return sizeof(Info); }

  /// \brief Returns the size of the type source info data block.
  unsigned getFullDataSize() const {
    return getLocalDataSize() + getPointeeLoc().getFullDataSize();
  }

  static bool classof(const TypeLoc *TL);
  static bool classof(const BlockPointerLoc *TL) { return true; }
};

/// \brief Wrapper for source info for member pointers.
class MemberPointerLoc : public DeclaratorLoc {
  struct Info {
    SourceLocation StarLoc;
  };

public:
  SourceLocation getStarLoc() const {
    return static_cast<Info*>(Data)->StarLoc;
  }
  void setStarLoc(SourceLocation Loc) {
    static_cast<Info*>(Data)->StarLoc = Loc;
  }

  TypeLoc getPointeeLoc() const {
    void *Next = static_cast<char*>(Data) + getLocalDataSize();
    return Create(cast<MemberPointerType>(Ty)->getPointeeType(), Next);
  }

  /// \brief Find the TypeSpecLoc that is part of this MemberPointerLoc.
  TypeSpecLoc getTypeSpecLoc() const {
    return getPointeeLoc().getTypeSpecLoc();
  }

  SourceRange getSourceRange() const {
    return SourceRange(getStarLoc(), getStarLoc());
  }

  /// \brief Returns the size of the type source info data block that is
  /// specific to this type.
  unsigned getLocalDataSize() const { return sizeof(Info); }

  /// \brief Returns the size of the type source info data block.
  unsigned getFullDataSize() const {
    return getLocalDataSize() + getPointeeLoc().getFullDataSize();
  }

  static bool classof(const TypeLoc *TL);
  static bool classof(const MemberPointerLoc *TL) { return true; }
};

/// \brief Wrapper for source info for references.
class ReferenceLoc : public DeclaratorLoc {
  struct Info {
    SourceLocation AmpLoc;
  };

public:
  SourceLocation getAmpLoc() const {
    return static_cast<Info*>(Data)->AmpLoc;
  }
  void setAmpLoc(SourceLocation Loc) {
    static_cast<Info*>(Data)->AmpLoc = Loc;
  }

  TypeLoc getPointeeLoc() const {
    void *Next = static_cast<char*>(Data) + getLocalDataSize();
    return Create(cast<ReferenceType>(Ty)->getPointeeType(), Next);
  }

  /// \brief Find the TypeSpecLoc that is part of this ReferenceLoc.
  TypeSpecLoc getTypeSpecLoc() const {
    return getPointeeLoc().getTypeSpecLoc();
  }

  SourceRange getSourceRange() const {
    return SourceRange(getAmpLoc(), getAmpLoc());
  }

  /// \brief Returns the size of the type source info data block that is
  /// specific to this type.
  unsigned getLocalDataSize() const { return sizeof(Info); }

  /// \brief Returns the size of the type source info data block.
  unsigned getFullDataSize() const {
    return getLocalDataSize() + getPointeeLoc().getFullDataSize();
  }

  static bool classof(const TypeLoc *TL);
  static bool classof(const ReferenceLoc *TL) { return true; }
};

/// \brief Wrapper for source info for functions.
class FunctionLoc : public DeclaratorLoc {
  struct Info {
    SourceLocation LParenLoc, RParenLoc;
  };
  // ParmVarDecls* are stored after Info, one for each argument.
  ParmVarDecl **getParmArray() const {
    return reinterpret_cast<ParmVarDecl**>(static_cast<Info*>(Data) + 1);
  }

public:
  SourceLocation getLParenLoc() const {
    return static_cast<Info*>(Data)->LParenLoc;
  }
  void setLParenLoc(SourceLocation Loc) {
    static_cast<Info*>(Data)->LParenLoc = Loc;
  }

  SourceLocation getRParenLoc() const {
    return static_cast<Info*>(Data)->RParenLoc;
  }
  void setRParenLoc(SourceLocation Loc) {
    static_cast<Info*>(Data)->RParenLoc = Loc;
  }

  unsigned getNumArgs() const {
    if (isa<FunctionNoProtoType>(Ty))
      return 0;
    return cast<FunctionProtoType>(Ty)->getNumArgs();
  }
  ParmVarDecl *getArg(unsigned i) const { return getParmArray()[i]; }
  void setArg(unsigned i, ParmVarDecl *VD) { getParmArray()[i] = VD; }

  TypeLoc getArgLoc(unsigned i) const;

  TypeLoc getResultLoc() const {
    void *Next = static_cast<char*>(Data) + getLocalDataSize();
    return Create(cast<FunctionType>(Ty)->getResultType(), Next);
  }

  /// \brief Find the TypeSpecLoc that is part of this FunctionLoc.
  TypeSpecLoc getTypeSpecLoc() const {
    return getResultLoc().getTypeSpecLoc();
  }
  SourceRange getSourceRange() const {
    return SourceRange(getLParenLoc(), getRParenLoc());
  }

  /// \brief Returns the size of the type source info data block that is
  /// specific to this type.
  unsigned getLocalDataSize() const {
    return sizeof(Info) + getNumArgs() * sizeof(ParmVarDecl*);
  }

  /// \brief Returns the size of the type source info data block.
  unsigned getFullDataSize() const {
    return getLocalDataSize() + getResultLoc().getFullDataSize();
  }

  static bool classof(const TypeLoc *TL);
  static bool classof(const FunctionLoc *TL) { return true; }
};

/// \brief Wrapper for source info for arrays.
class ArrayLoc : public DeclaratorLoc {
  struct Info {
    SourceLocation LBracketLoc, RBracketLoc;
    Expr *Size;
  };
public:
  SourceLocation getLBracketLoc() const {
    return static_cast<Info*>(Data)->LBracketLoc;
  }
  void setLBracketLoc(SourceLocation Loc) {
    static_cast<Info*>(Data)->LBracketLoc = Loc;
  }

  SourceLocation getRBracketLoc() const {
    return static_cast<Info*>(Data)->RBracketLoc;
  }
  void setRBracketLoc(SourceLocation Loc) {
    static_cast<Info*>(Data)->RBracketLoc = Loc;
  }

  Expr *getSizeExpr() const {
    return static_cast<Info*>(Data)->Size;
  }
  void setSizeExpr(Expr *Size) {
    static_cast<Info*>(Data)->Size = Size;
  }

  TypeLoc getElementLoc() const {
    void *Next = static_cast<char*>(Data) + getLocalDataSize();
    return Create(cast<ArrayType>(Ty)->getElementType(), Next);
  }

  /// \brief Find the TypeSpecLoc that is part of this ArrayLoc.
  TypeSpecLoc getTypeSpecLoc() const {
    return getElementLoc().getTypeSpecLoc();
  }
  SourceRange getSourceRange() const {
    return SourceRange(getLBracketLoc(), getRBracketLoc());
  }

  /// \brief Returns the size of the type source info data block that is
  /// specific to this type.
  unsigned getLocalDataSize() const { return sizeof(Info); }

  /// \brief Returns the size of the type source info data block.
  unsigned getFullDataSize() const {
    return getLocalDataSize() + getElementLoc().getFullDataSize();
  }

  static bool classof(const TypeLoc *TL);
  static bool classof(const ArrayLoc *TL) { return true; }
};

#define DISPATCH(CLASS) \
  return static_cast<ImplClass*>(this)->Visit ## CLASS(cast<CLASS>(TyLoc))

template<typename ImplClass, typename RetTy=void>
class TypeLocVisitor {
  class TypeDispatch : public TypeVisitor<TypeDispatch, RetTy> {
    ImplClass *Impl;
    TypeLoc TyLoc;

  public:
    TypeDispatch(ImplClass *impl, TypeLoc &tyLoc) : Impl(impl), TyLoc(tyLoc) { }
#define ABSTRACT_TYPELOC(CLASS)
#define TYPELOC(CLASS, PARENT, TYPE)                              \
    RetTy Visit##TYPE(TYPE *) {                                   \
      return Impl->Visit##CLASS(reinterpret_cast<CLASS&>(TyLoc)); \
    } 
#include "clang/AST/TypeLocNodes.def"
  };

public:
  RetTy Visit(TypeLoc TyLoc) {
    TypeDispatch TD(static_cast<ImplClass*>(this), TyLoc);
    return TD.Visit(TyLoc.getSourceType().getTypePtr());
  }

#define TYPELOC(CLASS, PARENT, TYPE) RetTy Visit##CLASS(CLASS TyLoc) {       \
  DISPATCH(PARENT);                                                          \
}
#include "clang/AST/TypeLocNodes.def"

  RetTy VisitTypeLoc(TypeLoc TyLoc) { return RetTy(); }
};

#undef DISPATCH

}

#endif
