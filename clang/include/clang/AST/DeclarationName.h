//===- DeclarationName.h - Representation of declaration names --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the DeclarationName and DeclarationNameTable classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLARATIONNAME_H
#define LLVM_CLANG_AST_DECLARATIONNAME_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/type_traits.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>

namespace clang {

class ASTContext;
template <typename> class CanQual;
class CXXDeductionGuideNameExtra;
class CXXLiteralOperatorIdName;
class CXXOperatorIdName;
class CXXSpecialName;
class DeclarationNameExtra;
class IdentifierInfo;
class MultiKeywordSelector;
enum OverloadedOperatorKind : int;
struct PrintingPolicy;
class QualType;
class TemplateDecl;
class Type;
class TypeSourceInfo;
class UsingDirectiveDecl;

using CanQualType = CanQual<Type>;

/// DeclarationName - The name of a declaration. In the common case,
/// this just stores an IdentifierInfo pointer to a normal
/// name. However, it also provides encodings for Objective-C
/// selectors (optimizing zero- and one-argument selectors, which make
/// up 78% percent of all selectors in Cocoa.h) and special C++ names
/// for constructors, destructors, and conversion functions.
class DeclarationName {
public:
  /// NameKind - The kind of name this object contains.
  enum NameKind {
    Identifier,
    ObjCZeroArgSelector,
    ObjCOneArgSelector,
    ObjCMultiArgSelector,
    CXXConstructorName,
    CXXDestructorName,
    CXXConversionFunctionName,
    CXXDeductionGuideName,
    CXXOperatorName,
    CXXLiteralOperatorName,
    CXXUsingDirective
  };

  static const unsigned NumNameKinds = CXXUsingDirective + 1;

private:
  friend class DeclarationNameTable;
  friend class NamedDecl;

  /// StoredNameKind - The kind of name that is actually stored in the
  /// upper bits of the Ptr field. This is only used internally.
  ///
  /// Note: The entries here are synchronized with the entries in Selector,
  /// for efficient translation between the two.
  enum StoredNameKind {
    StoredIdentifier = 0,
    StoredObjCZeroArgSelector = 0x01,
    StoredObjCOneArgSelector = 0x02,
    StoredDeclarationNameExtra = 0x03,
    PtrMask = 0x03
  };

  /// Ptr - The lowest two bits are used to express what kind of name
  /// we're actually storing, using the values of NameKind. Depending
  /// on the kind of name this is, the upper bits of Ptr may have one
  /// of several different meanings:
  ///
  ///   StoredIdentifier - The name is a normal identifier, and Ptr is
  ///   a normal IdentifierInfo pointer.
  ///
  ///   StoredObjCZeroArgSelector - The name is an Objective-C
  ///   selector with zero arguments, and Ptr is an IdentifierInfo
  ///   pointer pointing to the selector name.
  ///
  ///   StoredObjCOneArgSelector - The name is an Objective-C selector
  ///   with one argument, and Ptr is an IdentifierInfo pointer
  ///   pointing to the selector name.
  ///
  ///   StoredDeclarationNameExtra - Ptr is actually a pointer to a
  ///   DeclarationNameExtra structure, whose first value will tell us
  ///   whether this is an Objective-C selector, C++ operator-id name,
  ///   or special C++ name.
  uintptr_t Ptr = 0;

  // Construct a declaration name from the name of a C++ constructor,
  // destructor, or conversion function.
  DeclarationName(DeclarationNameExtra *Name)
      : Ptr(reinterpret_cast<uintptr_t>(Name)) {
    assert((Ptr & PtrMask) == 0 && "Improperly aligned DeclarationNameExtra");
    Ptr |= StoredDeclarationNameExtra;
  }

  /// Construct a declaration name from a raw pointer.
  DeclarationName(uintptr_t Ptr) : Ptr(Ptr) {}

  /// getStoredNameKind - Return the kind of object that is stored in
  /// Ptr.
  StoredNameKind getStoredNameKind() const {
    return static_cast<StoredNameKind>(Ptr & PtrMask);
  }

  /// getExtra - Get the "extra" information associated with this
  /// multi-argument selector or C++ special name.
  DeclarationNameExtra *getExtra() const {
    assert(getStoredNameKind() == StoredDeclarationNameExtra &&
           "Declaration name does not store an Extra structure");
    return reinterpret_cast<DeclarationNameExtra *>(Ptr & ~PtrMask);
  }

  /// getAsCXXSpecialName - If the stored pointer is actually a
  /// CXXSpecialName, returns a pointer to it. Otherwise, returns
  /// a NULL pointer.
  CXXSpecialName *getAsCXXSpecialName() const {
    NameKind Kind = getNameKind();
    if (Kind >= CXXConstructorName && Kind <= CXXConversionFunctionName)
      return reinterpret_cast<CXXSpecialName *>(getExtra());
    return nullptr;
  }

  /// If the stored pointer is actually a CXXDeductionGuideNameExtra, returns a
  /// pointer to it. Otherwise, returns a NULL pointer.
  CXXDeductionGuideNameExtra *getAsCXXDeductionGuideNameExtra() const {
    if (getNameKind() == CXXDeductionGuideName)
      return reinterpret_cast<CXXDeductionGuideNameExtra *>(getExtra());
    return nullptr;
  }

  /// getAsCXXOperatorIdName
  CXXOperatorIdName *getAsCXXOperatorIdName() const {
    if (getNameKind() == CXXOperatorName)
      return reinterpret_cast<CXXOperatorIdName *>(getExtra());
    return nullptr;
  }

  CXXLiteralOperatorIdName *getAsCXXLiteralOperatorIdName() const {
    if (getNameKind() == CXXLiteralOperatorName)
      return reinterpret_cast<CXXLiteralOperatorIdName *>(getExtra());
    return nullptr;
  }

  /// getFETokenInfoAsVoidSlow - Retrieves the front end-specified pointer
  /// for this name as a void pointer if it's not an identifier.
  void *getFETokenInfoAsVoidSlow() const;

public:
  /// DeclarationName - Used to create an empty selector.
  DeclarationName() = default;

  // Construct a declaration name from an IdentifierInfo *.
  DeclarationName(const IdentifierInfo *II)
      : Ptr(reinterpret_cast<uintptr_t>(II)) {
    assert((Ptr & PtrMask) == 0 && "Improperly aligned IdentifierInfo");
  }

  // Construct a declaration name from an Objective-C selector.
  DeclarationName(Selector Sel) : Ptr(Sel.InfoPtr) {}

  /// getUsingDirectiveName - Return name for all using-directives.
  static DeclarationName getUsingDirectiveName();

  // operator bool() - Evaluates true when this declaration name is
  // non-empty.
  explicit operator bool() const {
    return ((Ptr & PtrMask) != 0) ||
           (reinterpret_cast<IdentifierInfo *>(Ptr & ~PtrMask));
  }

  /// Evaluates true when this declaration name is empty.
  bool isEmpty() const {
    return !*this;
  }

  /// Predicate functions for querying what type of name this is.
  bool isIdentifier() const { return getStoredNameKind() == StoredIdentifier; }
  bool isObjCZeroArgSelector() const {
    return getStoredNameKind() == StoredObjCZeroArgSelector;
  }
  bool isObjCOneArgSelector() const {
    return getStoredNameKind() == StoredObjCOneArgSelector;
  }

  /// getNameKind - Determine what kind of name this is.
  NameKind getNameKind() const;

  /// Determines whether the name itself is dependent, e.g., because it 
  /// involves a C++ type that is itself dependent.
  ///
  /// Note that this does not capture all of the notions of "dependent name",
  /// because an identifier can be a dependent name if it is used as the 
  /// callee in a call expression with dependent arguments.
  bool isDependentName() const;
  
  /// getNameAsString - Retrieve the human-readable string for this name.
  std::string getAsString() const;

  /// getAsIdentifierInfo - Retrieve the IdentifierInfo * stored in
  /// this declaration name, or NULL if this declaration name isn't a
  /// simple identifier.
  IdentifierInfo *getAsIdentifierInfo() const {
    if (isIdentifier())
      return reinterpret_cast<IdentifierInfo *>(Ptr);
    return nullptr;
  }

  /// getAsOpaqueInteger - Get the representation of this declaration
  /// name as an opaque integer.
  uintptr_t getAsOpaqueInteger() const { return Ptr; }

  /// getAsOpaquePtr - Get the representation of this declaration name as
  /// an opaque pointer.
  void *getAsOpaquePtr() const { return reinterpret_cast<void*>(Ptr); }

  static DeclarationName getFromOpaquePtr(void *P) {
    DeclarationName N;
    N.Ptr = reinterpret_cast<uintptr_t> (P);
    return N;
  }

  static DeclarationName getFromOpaqueInteger(uintptr_t P) {
    DeclarationName N;
    N.Ptr = P;
    return N;
  }

  /// getCXXNameType - If this name is one of the C++ names (of a
  /// constructor, destructor, or conversion function), return the
  /// type associated with that name.
  QualType getCXXNameType() const;

  /// If this name is the name of a C++ deduction guide, return the
  /// template associated with that name.
  TemplateDecl *getCXXDeductionGuideTemplate() const;

  /// getCXXOverloadedOperator - If this name is the name of an
  /// overloadable operator in C++ (e.g., @c operator+), retrieve the
  /// kind of overloaded operator.
  OverloadedOperatorKind getCXXOverloadedOperator() const;

  /// getCXXLiteralIdentifier - If this name is the name of a literal
  /// operator, retrieve the identifier associated with it.
  IdentifierInfo *getCXXLiteralIdentifier() const;

  /// getObjCSelector - Get the Objective-C selector stored in this
  /// declaration name.
  Selector getObjCSelector() const {
    assert((getNameKind() == ObjCZeroArgSelector ||
            getNameKind() == ObjCOneArgSelector ||
            getNameKind() == ObjCMultiArgSelector ||
            Ptr == 0) && "Not a selector!");
    return Selector(Ptr);
  }

  /// getFETokenInfo/setFETokenInfo - The language front-end is
  /// allowed to associate arbitrary metadata with some kinds of
  /// declaration names, including normal identifiers and C++
  /// constructors, destructors, and conversion functions.
  template<typename T>
  T *getFETokenInfo() const {
    if (const IdentifierInfo *Info = getAsIdentifierInfo())
      return Info->getFETokenInfo<T>();
    return static_cast<T*>(getFETokenInfoAsVoidSlow());
  }

  void setFETokenInfo(void *T);

  /// operator== - Determine whether the specified names are identical..
  friend bool operator==(DeclarationName LHS, DeclarationName RHS) {
    return LHS.Ptr == RHS.Ptr;
  }

  /// operator!= - Determine whether the specified names are different.
  friend bool operator!=(DeclarationName LHS, DeclarationName RHS) {
    return LHS.Ptr != RHS.Ptr;
  }

  static DeclarationName getEmptyMarker() {
    return DeclarationName(uintptr_t(-1));
  }

  static DeclarationName getTombstoneMarker() {
    return DeclarationName(uintptr_t(-2));
  }

  static int compare(DeclarationName LHS, DeclarationName RHS);

  void print(raw_ostream &OS, const PrintingPolicy &Policy);

  void dump() const;
};

raw_ostream &operator<<(raw_ostream &OS, DeclarationName N);

/// Ordering on two declaration names. If both names are identifiers,
/// this provides a lexicographical ordering.
inline bool operator<(DeclarationName LHS, DeclarationName RHS) {
  return DeclarationName::compare(LHS, RHS) < 0;
}

/// Ordering on two declaration names. If both names are identifiers,
/// this provides a lexicographical ordering.
inline bool operator>(DeclarationName LHS, DeclarationName RHS) {
  return DeclarationName::compare(LHS, RHS) > 0;
}

/// Ordering on two declaration names. If both names are identifiers,
/// this provides a lexicographical ordering.
inline bool operator<=(DeclarationName LHS, DeclarationName RHS) {
  return DeclarationName::compare(LHS, RHS) <= 0;
}

/// Ordering on two declaration names. If both names are identifiers,
/// this provides a lexicographical ordering.
inline bool operator>=(DeclarationName LHS, DeclarationName RHS) {
  return DeclarationName::compare(LHS, RHS) >= 0;
}

/// DeclarationNameTable - Used to store and retrieve DeclarationName
/// instances for the various kinds of declaration names, e.g., normal
/// identifiers, C++ constructor names, etc. This class contains
/// uniqued versions of each of the C++ special names, which can be
/// retrieved using its member functions (e.g.,
/// getCXXConstructorName).
class DeclarationNameTable {
  const ASTContext &Ctx;

  // Actually a FoldingSet<CXXSpecialName> *
  void *CXXSpecialNamesImpl;

  // Operator names
  CXXOperatorIdName *CXXOperatorNames;

  // Actually a CXXOperatorIdName*
  void *CXXLiteralOperatorNames;

  // FoldingSet<CXXDeductionGuideNameExtra> *
  void *CXXDeductionGuideNames;

public:
  DeclarationNameTable(const ASTContext &C);
  DeclarationNameTable(const DeclarationNameTable &) = delete;
  DeclarationNameTable &operator=(const DeclarationNameTable &) = delete;

  ~DeclarationNameTable();

  /// getIdentifier - Create a declaration name that is a simple
  /// identifier.
  DeclarationName getIdentifier(const IdentifierInfo *ID) {
    return DeclarationName(ID);
  }

  /// getCXXConstructorName - Returns the name of a C++ constructor
  /// for the given Type.
  DeclarationName getCXXConstructorName(CanQualType Ty);

  /// getCXXDestructorName - Returns the name of a C++ destructor
  /// for the given Type.
  DeclarationName getCXXDestructorName(CanQualType Ty);

  /// Returns the name of a C++ deduction guide for the given template.
  DeclarationName getCXXDeductionGuideName(TemplateDecl *TD);

  /// getCXXConversionFunctionName - Returns the name of a C++
  /// conversion function for the given Type.
  DeclarationName getCXXConversionFunctionName(CanQualType Ty);

  /// getCXXSpecialName - Returns a declaration name for special kind
  /// of C++ name, e.g., for a constructor, destructor, or conversion
  /// function.
  DeclarationName getCXXSpecialName(DeclarationName::NameKind Kind,
                                    CanQualType Ty);

  /// getCXXOperatorName - Get the name of the overloadable C++
  /// operator corresponding to Op.
  DeclarationName getCXXOperatorName(OverloadedOperatorKind Op);

  /// getCXXLiteralOperatorName - Get the name of the literal operator function
  /// with II as the identifier.
  DeclarationName getCXXLiteralOperatorName(IdentifierInfo *II);
};

/// DeclarationNameLoc - Additional source/type location info
/// for a declaration name. Needs a DeclarationName in order
/// to be interpreted correctly.
struct DeclarationNameLoc {
  // The source location for identifier stored elsewhere.
  // struct {} Identifier;

  // Type info for constructors, destructors and conversion functions.
  // Locations (if any) for the tilde (destructor) or operator keyword
  // (conversion) are stored elsewhere.
  struct NT {
    TypeSourceInfo *TInfo;
  };

  // The location (if any) of the operator keyword is stored elsewhere.
  struct CXXOpName {
    unsigned BeginOpNameLoc;
    unsigned EndOpNameLoc;
  };

  // The location (if any) of the operator keyword is stored elsewhere.
  struct CXXLitOpName {
    unsigned OpNameLoc;
  };

  // struct {} CXXUsingDirective;
  // struct {} ObjCZeroArgSelector;
  // struct {} ObjCOneArgSelector;
  // struct {} ObjCMultiArgSelector;
  union {
    struct NT NamedType;
    struct CXXOpName CXXOperatorName;
    struct CXXLitOpName CXXLiteralOperatorName;
  };

  DeclarationNameLoc(DeclarationName Name);

  // FIXME: this should go away once all DNLocs are properly initialized.
  DeclarationNameLoc() { memset((void*) this, 0, sizeof(*this)); }
};

/// DeclarationNameInfo - A collector data type for bundling together
/// a DeclarationName and the correspnding source/type location info.
struct DeclarationNameInfo {
private:
  /// Name - The declaration name, also encoding name kind.
  DeclarationName Name;

  /// Loc - The main source location for the declaration name.
  SourceLocation NameLoc;

  /// Info - Further source/type location info for special kinds of names.
  DeclarationNameLoc LocInfo;

public:
  // FIXME: remove it.
  DeclarationNameInfo() = default;

  DeclarationNameInfo(DeclarationName Name, SourceLocation NameLoc)
      : Name(Name), NameLoc(NameLoc), LocInfo(Name) {}

  DeclarationNameInfo(DeclarationName Name, SourceLocation NameLoc,
                      DeclarationNameLoc LocInfo)
      : Name(Name), NameLoc(NameLoc), LocInfo(LocInfo) {}

  /// getName - Returns the embedded declaration name.
  DeclarationName getName() const { return Name; }

  /// setName - Sets the embedded declaration name.
  void setName(DeclarationName N) { Name = N; }

  /// getLoc - Returns the main location of the declaration name.
  SourceLocation getLoc() const { return NameLoc; }

  /// setLoc - Sets the main location of the declaration name.
  void setLoc(SourceLocation L) { NameLoc = L; }

  const DeclarationNameLoc &getInfo() const { return LocInfo; }
  DeclarationNameLoc &getInfo() { return LocInfo; }
  void setInfo(const DeclarationNameLoc &Info) { LocInfo = Info; }

  /// getNamedTypeInfo - Returns the source type info associated to
  /// the name. Assumes it is a constructor, destructor or conversion.
  TypeSourceInfo *getNamedTypeInfo() const {
    assert(Name.getNameKind() == DeclarationName::CXXConstructorName ||
           Name.getNameKind() == DeclarationName::CXXDestructorName ||
           Name.getNameKind() == DeclarationName::CXXConversionFunctionName);
    return LocInfo.NamedType.TInfo;
  }

  /// setNamedTypeInfo - Sets the source type info associated to
  /// the name. Assumes it is a constructor, destructor or conversion.
  void setNamedTypeInfo(TypeSourceInfo *TInfo) {
    assert(Name.getNameKind() == DeclarationName::CXXConstructorName ||
           Name.getNameKind() == DeclarationName::CXXDestructorName ||
           Name.getNameKind() == DeclarationName::CXXConversionFunctionName);
    LocInfo.NamedType.TInfo = TInfo;
  }

  /// getCXXOperatorNameRange - Gets the range of the operator name
  /// (without the operator keyword). Assumes it is a (non-literal) operator.
  SourceRange getCXXOperatorNameRange() const {
    assert(Name.getNameKind() == DeclarationName::CXXOperatorName);
    return SourceRange(
     SourceLocation::getFromRawEncoding(LocInfo.CXXOperatorName.BeginOpNameLoc),
     SourceLocation::getFromRawEncoding(LocInfo.CXXOperatorName.EndOpNameLoc)
                       );
  }

  /// setCXXOperatorNameRange - Sets the range of the operator name
  /// (without the operator keyword). Assumes it is a C++ operator.
  void setCXXOperatorNameRange(SourceRange R) {
    assert(Name.getNameKind() == DeclarationName::CXXOperatorName);
    LocInfo.CXXOperatorName.BeginOpNameLoc = R.getBegin().getRawEncoding();
    LocInfo.CXXOperatorName.EndOpNameLoc = R.getEnd().getRawEncoding();
  }

  /// getCXXLiteralOperatorNameLoc - Returns the location of the literal
  /// operator name (not the operator keyword).
  /// Assumes it is a literal operator.
  SourceLocation getCXXLiteralOperatorNameLoc() const {
    assert(Name.getNameKind() == DeclarationName::CXXLiteralOperatorName);
    return SourceLocation::
      getFromRawEncoding(LocInfo.CXXLiteralOperatorName.OpNameLoc);
  }

  /// setCXXLiteralOperatorNameLoc - Sets the location of the literal
  /// operator name (not the operator keyword).
  /// Assumes it is a literal operator.
  void setCXXLiteralOperatorNameLoc(SourceLocation Loc) {
    assert(Name.getNameKind() == DeclarationName::CXXLiteralOperatorName);
    LocInfo.CXXLiteralOperatorName.OpNameLoc = Loc.getRawEncoding();
  }

  /// Determine whether this name involves a template parameter.
  bool isInstantiationDependent() const;
  
  /// Determine whether this name contains an unexpanded
  /// parameter pack.
  bool containsUnexpandedParameterPack() const;

  /// getAsString - Retrieve the human-readable string for this name.
  std::string getAsString() const;

  /// printName - Print the human-readable name to a stream.
  void printName(raw_ostream &OS) const;

  /// getBeginLoc - Retrieve the location of the first token.
  SourceLocation getBeginLoc() const { return NameLoc; }

  /// getEndLoc - Retrieve the location of the last token.
  SourceLocation getEndLoc() const;

  /// getSourceRange - The range of the declaration name.
  SourceRange getSourceRange() const LLVM_READONLY {
    return SourceRange(getLocStart(), getLocEnd());
  }

  SourceLocation getLocStart() const LLVM_READONLY {
    return getBeginLoc();
  }

  SourceLocation getLocEnd() const LLVM_READONLY {
    SourceLocation EndLoc = getEndLoc();
    return EndLoc.isValid() ? EndLoc : getLocStart();
  }
};

/// Insertion operator for diagnostics.  This allows sending DeclarationName's
/// into a diagnostic with <<.
inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           DeclarationName N) {
  DB.AddTaggedVal(N.getAsOpaqueInteger(),
                  DiagnosticsEngine::ak_declarationname);
  return DB;
}

/// Insertion operator for partial diagnostics.  This allows binding
/// DeclarationName's into a partial diagnostic with <<.
inline const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
                                           DeclarationName N) {
  PD.AddTaggedVal(N.getAsOpaqueInteger(),
                  DiagnosticsEngine::ak_declarationname);
  return PD;
}

inline raw_ostream &operator<<(raw_ostream &OS,
                                     DeclarationNameInfo DNInfo) {
  DNInfo.printName(OS);
  return OS;
}

} // namespace clang

namespace llvm {

/// Define DenseMapInfo so that DeclarationNames can be used as keys
/// in DenseMap and DenseSets.
template<>
struct DenseMapInfo<clang::DeclarationName> {
  static inline clang::DeclarationName getEmptyKey() {
    return clang::DeclarationName::getEmptyMarker();
  }

  static inline clang::DeclarationName getTombstoneKey() {
    return clang::DeclarationName::getTombstoneMarker();
  }

  static unsigned getHashValue(clang::DeclarationName Name) {
    return DenseMapInfo<void*>::getHashValue(Name.getAsOpaquePtr());
  }

  static inline bool
  isEqual(clang::DeclarationName LHS, clang::DeclarationName RHS) {
    return LHS == RHS;
  }
};

template <>
struct isPodLike<clang::DeclarationName> { static const bool value = true; };

} // namespace llvm

#endif // LLVM_CLANG_AST_DECLARATIONNAME_H
