//===--- DeclSpec.h - Parsed declaration specifiers -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file defines the classes used to store parsed information about
/// declaration-specifiers and declarators.
///
/// \verbatim
///   static const int volatile x, *y, *(*(*z)[10])(const void *x);
///   ------------------------- -  --  ---------------------------
///     declaration-specifiers  \  |   /
///                            declarators
/// \endverbatim
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_DECLSPEC_H
#define LLVM_CLANG_SEMA_DECLSPEC_H

#include "clang/Sema/AttributeList.h"
#include "clang/Sema/Ownership.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/Lex/Token.h"
#include "clang/Basic/ExceptionSpecificationType.h"
#include "clang/Basic/Lambda.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang {
  class ASTContext;
  class TypeLoc;
  class LangOptions;
  class DiagnosticsEngine;
  class IdentifierInfo;
  class NamespaceAliasDecl;
  class NamespaceDecl;
  class NestedNameSpecifier;
  class NestedNameSpecifierLoc;
  class ObjCDeclSpec;
  class Preprocessor;
  class Sema;
  class Declarator;
  struct TemplateIdAnnotation;

/// \brief Represents a C++ nested-name-specifier or a global scope specifier.
///
/// These can be in 3 states:
///   1) Not present, identified by isEmpty()
///   2) Present, identified by isNotEmpty()
///      2.a) Valid, idenified by isValid()
///      2.b) Invalid, identified by isInvalid().
///
/// isSet() is deprecated because it mostly corresponded to "valid" but was
/// often used as if it meant "present".
///
/// The actual scope is described by getScopeRep().
class CXXScopeSpec {
  SourceRange Range;  
  NestedNameSpecifierLocBuilder Builder;

public:
  const SourceRange &getRange() const { return Range; }
  void setRange(const SourceRange &R) { Range = R; }
  void setBeginLoc(SourceLocation Loc) { Range.setBegin(Loc); }
  void setEndLoc(SourceLocation Loc) { Range.setEnd(Loc); }
  SourceLocation getBeginLoc() const { return Range.getBegin(); }
  SourceLocation getEndLoc() const { return Range.getEnd(); }

  /// \brief Retrieve the representation of the nested-name-specifier.
  NestedNameSpecifier *getScopeRep() const { 
    return Builder.getRepresentation(); 
  }

  /// \brief Extend the current nested-name-specifier by another
  /// nested-name-specifier component of the form 'type::'.
  ///
  /// \param Context The AST context in which this nested-name-specifier
  /// resides.
  ///
  /// \param TemplateKWLoc The location of the 'template' keyword, if present.
  ///
  /// \param TL The TypeLoc that describes the type preceding the '::'.
  ///
  /// \param ColonColonLoc The location of the trailing '::'.
  void Extend(ASTContext &Context, SourceLocation TemplateKWLoc, TypeLoc TL,
              SourceLocation ColonColonLoc);

  /// \brief Extend the current nested-name-specifier by another 
  /// nested-name-specifier component of the form 'identifier::'.
  ///
  /// \param Context The AST context in which this nested-name-specifier
  /// resides.
  ///
  /// \param Identifier The identifier.
  ///
  /// \param IdentifierLoc The location of the identifier.
  ///
  /// \param ColonColonLoc The location of the trailing '::'.
  void Extend(ASTContext &Context, IdentifierInfo *Identifier,
              SourceLocation IdentifierLoc, SourceLocation ColonColonLoc);

  /// \brief Extend the current nested-name-specifier by another 
  /// nested-name-specifier component of the form 'namespace::'.
  ///
  /// \param Context The AST context in which this nested-name-specifier
  /// resides.
  ///
  /// \param Namespace The namespace.
  ///
  /// \param NamespaceLoc The location of the namespace name.
  ///
  /// \param ColonColonLoc The location of the trailing '::'.
  void Extend(ASTContext &Context, NamespaceDecl *Namespace,
              SourceLocation NamespaceLoc, SourceLocation ColonColonLoc);

  /// \brief Extend the current nested-name-specifier by another 
  /// nested-name-specifier component of the form 'namespace-alias::'.
  ///
  /// \param Context The AST context in which this nested-name-specifier
  /// resides.
  ///
  /// \param Alias The namespace alias.
  ///
  /// \param AliasLoc The location of the namespace alias 
  /// name.
  ///
  /// \param ColonColonLoc The location of the trailing '::'.
  void Extend(ASTContext &Context, NamespaceAliasDecl *Alias,
              SourceLocation AliasLoc, SourceLocation ColonColonLoc);

  /// \brief Turn this (empty) nested-name-specifier into the global
  /// nested-name-specifier '::'.
  void MakeGlobal(ASTContext &Context, SourceLocation ColonColonLoc);
  
  /// \brief Make a new nested-name-specifier from incomplete source-location
  /// information.
  ///
  /// FIXME: This routine should be used very, very rarely, in cases where we
  /// need to synthesize a nested-name-specifier. Most code should instead use
  /// \c Adopt() with a proper \c NestedNameSpecifierLoc.
  void MakeTrivial(ASTContext &Context, NestedNameSpecifier *Qualifier, 
                   SourceRange R);
  
  /// \brief Adopt an existing nested-name-specifier (with source-range 
  /// information).
  void Adopt(NestedNameSpecifierLoc Other);
  
  /// \brief Retrieve a nested-name-specifier with location information, copied
  /// into the given AST context.
  ///
  /// \param Context The context into which this nested-name-specifier will be
  /// copied.
  NestedNameSpecifierLoc getWithLocInContext(ASTContext &Context) const;

  /// \brief Retrieve the location of the name in the last qualifier
  /// in this nested name specifier.
  ///
  /// For example, the location of \c bar
  /// in
  /// \verbatim
  ///   \::foo::bar<0>::
  ///           ^~~
  /// \endverbatim
  SourceLocation getLastQualifierNameLoc() const;

  /// No scope specifier.
  bool isEmpty() const { return !Range.isValid(); }
  /// A scope specifier is present, but may be valid or invalid.
  bool isNotEmpty() const { return !isEmpty(); }

  /// An error occurred during parsing of the scope specifier.
  bool isInvalid() const { return isNotEmpty() && getScopeRep() == 0; }
  /// A scope specifier is present, and it refers to a real scope.
  bool isValid() const { return isNotEmpty() && getScopeRep() != 0; }

  /// \brief Indicate that this nested-name-specifier is invalid.
  void SetInvalid(SourceRange R) { 
    assert(R.isValid() && "Must have a valid source range");
    if (Range.getBegin().isInvalid())
      Range.setBegin(R.getBegin());
    Range.setEnd(R.getEnd());
    Builder.Clear();
  }
  
  /// Deprecated.  Some call sites intend isNotEmpty() while others intend
  /// isValid().
  bool isSet() const { return getScopeRep() != 0; }

  void clear() {
    Range = SourceRange();
    Builder.Clear();
  }

  /// \brief Retrieve the data associated with the source-location information.
  char *location_data() const { return Builder.getBuffer().first; }
  
  /// \brief Retrieve the size of the data associated with source-location 
  /// information.
  unsigned location_size() const { return Builder.getBuffer().second; }
};

/// \brief Captures information about "declaration specifiers".
///
/// "Declaration specifiers" encompasses storage-class-specifiers,
/// type-specifiers, type-qualifiers, and function-specifiers.
class DeclSpec {
public:
  /// \brief storage-class-specifier
  /// \note The order of these enumerators is important for diagnostics.
  enum SCS {
    SCS_unspecified = 0,
    SCS_typedef,
    SCS_extern,
    SCS_static,
    SCS_auto,
    SCS_register,
    SCS_private_extern,
    SCS_mutable
  };

  // Import type specifier width enumeration and constants.
  typedef TypeSpecifierWidth TSW;
  static const TSW TSW_unspecified = clang::TSW_unspecified;
  static const TSW TSW_short = clang::TSW_short;
  static const TSW TSW_long = clang::TSW_long;
  static const TSW TSW_longlong = clang::TSW_longlong;
  
  enum TSC {
    TSC_unspecified,
    TSC_imaginary,
    TSC_complex
  };

  // Import type specifier sign enumeration and constants.
  typedef TypeSpecifierSign TSS;
  static const TSS TSS_unspecified = clang::TSS_unspecified;
  static const TSS TSS_signed = clang::TSS_signed;
  static const TSS TSS_unsigned = clang::TSS_unsigned;

  // Import type specifier type enumeration and constants.
  typedef TypeSpecifierType TST;
  static const TST TST_unspecified = clang::TST_unspecified;
  static const TST TST_void = clang::TST_void;
  static const TST TST_char = clang::TST_char;
  static const TST TST_wchar = clang::TST_wchar;
  static const TST TST_char16 = clang::TST_char16;
  static const TST TST_char32 = clang::TST_char32;
  static const TST TST_int = clang::TST_int;
  static const TST TST_int128 = clang::TST_int128;
  static const TST TST_half = clang::TST_half;
  static const TST TST_float = clang::TST_float;
  static const TST TST_double = clang::TST_double;
  static const TST TST_bool = clang::TST_bool;
  static const TST TST_decimal32 = clang::TST_decimal32;
  static const TST TST_decimal64 = clang::TST_decimal64;
  static const TST TST_decimal128 = clang::TST_decimal128;
  static const TST TST_enum = clang::TST_enum;
  static const TST TST_union = clang::TST_union;
  static const TST TST_struct = clang::TST_struct;
  static const TST TST_interface = clang::TST_interface;
  static const TST TST_class = clang::TST_class;
  static const TST TST_typename = clang::TST_typename;
  static const TST TST_typeofType = clang::TST_typeofType;
  static const TST TST_typeofExpr = clang::TST_typeofExpr;
  static const TST TST_decltype = clang::TST_decltype;
  static const TST TST_underlyingType = clang::TST_underlyingType;
  static const TST TST_auto = clang::TST_auto;
  static const TST TST_unknown_anytype = clang::TST_unknown_anytype;
  static const TST TST_atomic = clang::TST_atomic;
  static const TST TST_error = clang::TST_error;

  // type-qualifiers
  enum TQ {   // NOTE: These flags must be kept in sync with Qualifiers::TQ.
    TQ_unspecified = 0,
    TQ_const       = 1,
    TQ_restrict    = 2,
    TQ_volatile    = 4
  };

  /// ParsedSpecifiers - Flags to query which specifiers were applied.  This is
  /// returned by getParsedSpecifiers.
  enum ParsedSpecifiers {
    PQ_None                  = 0,
    PQ_StorageClassSpecifier = 1,
    PQ_TypeSpecifier         = 2,
    PQ_TypeQualifier         = 4,
    PQ_FunctionSpecifier     = 8
  };

private:
  // storage-class-specifier
  /*SCS*/unsigned StorageClassSpec : 3;
  unsigned SCS_thread_specified : 1;
  unsigned SCS_extern_in_linkage_spec : 1;

  // type-specifier
  /*TSW*/unsigned TypeSpecWidth : 2;
  /*TSC*/unsigned TypeSpecComplex : 2;
  /*TSS*/unsigned TypeSpecSign : 2;
  /*TST*/unsigned TypeSpecType : 5;
  unsigned TypeAltiVecVector : 1;
  unsigned TypeAltiVecPixel : 1;
  unsigned TypeAltiVecBool : 1;
  unsigned TypeSpecOwned : 1;

  // type-qualifiers
  unsigned TypeQualifiers : 3;  // Bitwise OR of TQ.

  // function-specifier
  unsigned FS_inline_specified : 1;
  unsigned FS_virtual_specified : 1;
  unsigned FS_explicit_specified : 1;

  // friend-specifier
  unsigned Friend_specified : 1;

  // constexpr-specifier
  unsigned Constexpr_specified : 1;

  /*SCS*/unsigned StorageClassSpecAsWritten : 3;

  union {
    UnionParsedType TypeRep;
    Decl *DeclRep;
    Expr *ExprRep;
  };

  // attributes.
  ParsedAttributes Attrs;

  // Scope specifier for the type spec, if applicable.
  CXXScopeSpec TypeScope;

  // List of protocol qualifiers for objective-c classes.  Used for
  // protocol-qualified interfaces "NString<foo>" and protocol-qualified id
  // "id<foo>".
  Decl * const *ProtocolQualifiers;
  unsigned NumProtocolQualifiers;
  SourceLocation ProtocolLAngleLoc;
  SourceLocation *ProtocolLocs;

  // SourceLocation info.  These are null if the item wasn't specified or if
  // the setting was synthesized.
  SourceRange Range;

  SourceLocation StorageClassSpecLoc, SCS_threadLoc;
  SourceLocation TSWLoc, TSCLoc, TSSLoc, TSTLoc, AltiVecLoc;
  /// TSTNameLoc - If TypeSpecType is any of class, enum, struct, union,
  /// typename, then this is the location of the named type (if present);
  /// otherwise, it is the same as TSTLoc. Hence, the pair TSTLoc and
  /// TSTNameLoc provides source range info for tag types.
  SourceLocation TSTNameLoc;
  SourceRange TypeofParensRange;
  SourceLocation TQ_constLoc, TQ_restrictLoc, TQ_volatileLoc;
  SourceLocation FS_inlineLoc, FS_virtualLoc, FS_explicitLoc;
  SourceLocation FriendLoc, ModulePrivateLoc, ConstexprLoc;

  WrittenBuiltinSpecs writtenBS;
  void SaveWrittenBuiltinSpecs();
  void SaveStorageSpecifierAsWritten();

  ObjCDeclSpec *ObjCQualifiers;

  static bool isTypeRep(TST T) {
    return (T == TST_typename || T == TST_typeofType ||
            T == TST_underlyingType || T == TST_atomic);
  }
  static bool isExprRep(TST T) {
    return (T == TST_typeofExpr || T == TST_decltype);
  }
  static bool isDeclRep(TST T) {
    return (T == TST_enum || T == TST_struct ||
            T == TST_interface || T == TST_union ||
            T == TST_class);
  }

  DeclSpec(const DeclSpec &) LLVM_DELETED_FUNCTION;
  void operator=(const DeclSpec &) LLVM_DELETED_FUNCTION;
public:

  DeclSpec(AttributeFactory &attrFactory)
    : StorageClassSpec(SCS_unspecified),
      SCS_thread_specified(false),
      SCS_extern_in_linkage_spec(false),
      TypeSpecWidth(TSW_unspecified),
      TypeSpecComplex(TSC_unspecified),
      TypeSpecSign(TSS_unspecified),
      TypeSpecType(TST_unspecified),
      TypeAltiVecVector(false),
      TypeAltiVecPixel(false),
      TypeAltiVecBool(false),
      TypeSpecOwned(false),
      TypeQualifiers(TQ_unspecified),
      FS_inline_specified(false),
      FS_virtual_specified(false),
      FS_explicit_specified(false),
      Friend_specified(false),
      Constexpr_specified(false),
      StorageClassSpecAsWritten(SCS_unspecified),
      Attrs(attrFactory),
      ProtocolQualifiers(0),
      NumProtocolQualifiers(0),
      ProtocolLocs(0),
      writtenBS(),
      ObjCQualifiers(0) {
  }
  ~DeclSpec() {
    delete [] ProtocolQualifiers;
    delete [] ProtocolLocs;
  }
  // storage-class-specifier
  SCS getStorageClassSpec() const { return (SCS)StorageClassSpec; }
  bool isThreadSpecified() const { return SCS_thread_specified; }
  bool isExternInLinkageSpec() const { return SCS_extern_in_linkage_spec; }
  void setExternInLinkageSpec(bool Value) {
    SCS_extern_in_linkage_spec = Value;
  }

  SourceLocation getStorageClassSpecLoc() const { return StorageClassSpecLoc; }
  SourceLocation getThreadSpecLoc() const { return SCS_threadLoc; }

  void ClearStorageClassSpecs() {
    StorageClassSpec     = DeclSpec::SCS_unspecified;
    SCS_thread_specified = false;
    SCS_extern_in_linkage_spec = false;
    StorageClassSpecLoc  = SourceLocation();
    SCS_threadLoc        = SourceLocation();
  }

  // type-specifier
  TSW getTypeSpecWidth() const { return (TSW)TypeSpecWidth; }
  TSC getTypeSpecComplex() const { return (TSC)TypeSpecComplex; }
  TSS getTypeSpecSign() const { return (TSS)TypeSpecSign; }
  TST getTypeSpecType() const { return (TST)TypeSpecType; }
  bool isTypeAltiVecVector() const { return TypeAltiVecVector; }
  bool isTypeAltiVecPixel() const { return TypeAltiVecPixel; }
  bool isTypeAltiVecBool() const { return TypeAltiVecBool; }
  bool isTypeSpecOwned() const { return TypeSpecOwned; }
  ParsedType getRepAsType() const {
    assert(isTypeRep((TST) TypeSpecType) && "DeclSpec does not store a type");
    return TypeRep;
  }
  Decl *getRepAsDecl() const {
    assert(isDeclRep((TST) TypeSpecType) && "DeclSpec does not store a decl");
    return DeclRep;
  }
  Expr *getRepAsExpr() const {
    assert(isExprRep((TST) TypeSpecType) && "DeclSpec does not store an expr");
    return ExprRep;
  }
  CXXScopeSpec &getTypeSpecScope() { return TypeScope; }
  const CXXScopeSpec &getTypeSpecScope() const { return TypeScope; }

  const SourceRange &getSourceRange() const LLVM_READONLY { return Range; }
  SourceLocation getLocStart() const LLVM_READONLY { return Range.getBegin(); }
  SourceLocation getLocEnd() const LLVM_READONLY { return Range.getEnd(); }

  SourceLocation getTypeSpecWidthLoc() const { return TSWLoc; }
  SourceLocation getTypeSpecComplexLoc() const { return TSCLoc; }
  SourceLocation getTypeSpecSignLoc() const { return TSSLoc; }
  SourceLocation getTypeSpecTypeLoc() const { return TSTLoc; }
  SourceLocation getAltiVecLoc() const { return AltiVecLoc; }

  SourceLocation getTypeSpecTypeNameLoc() const {
    assert(isDeclRep((TST) TypeSpecType) || TypeSpecType == TST_typename);
    return TSTNameLoc;
  }

  SourceRange getTypeofParensRange() const { return TypeofParensRange; }
  void setTypeofParensRange(SourceRange range) { TypeofParensRange = range; }

  /// \brief Turn a type-specifier-type into a string like "_Bool" or "union".
  static const char *getSpecifierName(DeclSpec::TST T);
  static const char *getSpecifierName(DeclSpec::TQ Q);
  static const char *getSpecifierName(DeclSpec::TSS S);
  static const char *getSpecifierName(DeclSpec::TSC C);
  static const char *getSpecifierName(DeclSpec::TSW W);
  static const char *getSpecifierName(DeclSpec::SCS S);

  // type-qualifiers

  /// getTypeQualifiers - Return a set of TQs.
  unsigned getTypeQualifiers() const { return TypeQualifiers; }
  SourceLocation getConstSpecLoc() const { return TQ_constLoc; }
  SourceLocation getRestrictSpecLoc() const { return TQ_restrictLoc; }
  SourceLocation getVolatileSpecLoc() const { return TQ_volatileLoc; }

  /// \brief Clear out all of the type qualifiers.
  void ClearTypeQualifiers() {
    TypeQualifiers = 0;
    TQ_constLoc = SourceLocation();
    TQ_restrictLoc = SourceLocation();
    TQ_volatileLoc = SourceLocation();
  }

  // function-specifier
  bool isInlineSpecified() const { return FS_inline_specified; }
  SourceLocation getInlineSpecLoc() const { return FS_inlineLoc; }

  bool isVirtualSpecified() const { return FS_virtual_specified; }
  SourceLocation getVirtualSpecLoc() const { return FS_virtualLoc; }

  bool isExplicitSpecified() const { return FS_explicit_specified; }
  SourceLocation getExplicitSpecLoc() const { return FS_explicitLoc; }

  void ClearFunctionSpecs() {
    FS_inline_specified = false;
    FS_inlineLoc = SourceLocation();
    FS_virtual_specified = false;
    FS_virtualLoc = SourceLocation();
    FS_explicit_specified = false;
    FS_explicitLoc = SourceLocation();
  }

  /// \brief Return true if any type-specifier has been found.
  bool hasTypeSpecifier() const {
    return getTypeSpecType() != DeclSpec::TST_unspecified ||
           getTypeSpecWidth() != DeclSpec::TSW_unspecified ||
           getTypeSpecComplex() != DeclSpec::TSC_unspecified ||
           getTypeSpecSign() != DeclSpec::TSS_unspecified;
  }

  /// \brief Return a bitmask of which flavors of specifiers this
  /// DeclSpec includes.
  unsigned getParsedSpecifiers() const;

  SCS getStorageClassSpecAsWritten() const {
    return (SCS)StorageClassSpecAsWritten;
  }

  /// isEmpty - Return true if this declaration specifier is completely empty:
  /// no tokens were parsed in the production of it.
  bool isEmpty() const {
    return getParsedSpecifiers() == DeclSpec::PQ_None;
  }

  void SetRangeStart(SourceLocation Loc) { Range.setBegin(Loc); }
  void SetRangeEnd(SourceLocation Loc) { Range.setEnd(Loc); }

  /// These methods set the specified attribute of the DeclSpec and
  /// return false if there was no error.  If an error occurs (for
  /// example, if we tried to set "auto" on a spec with "extern"
  /// already set), they return true and set PrevSpec and DiagID
  /// such that
  ///   Diag(Loc, DiagID) << PrevSpec;
  /// will yield a useful result.
  ///
  /// TODO: use a more general approach that still allows these
  /// diagnostics to be ignored when desired.
  bool SetStorageClassSpec(Sema &S, SCS SC, SourceLocation Loc,
                           const char *&PrevSpec, unsigned &DiagID);
  bool SetStorageClassSpecThread(SourceLocation Loc, const char *&PrevSpec,
                                 unsigned &DiagID);
  bool SetTypeSpecWidth(TSW W, SourceLocation Loc, const char *&PrevSpec,
                        unsigned &DiagID);
  bool SetTypeSpecComplex(TSC C, SourceLocation Loc, const char *&PrevSpec,
                          unsigned &DiagID);
  bool SetTypeSpecSign(TSS S, SourceLocation Loc, const char *&PrevSpec,
                       unsigned &DiagID);
  bool SetTypeSpecType(TST T, SourceLocation Loc, const char *&PrevSpec,
                       unsigned &DiagID);
  bool SetTypeSpecType(TST T, SourceLocation Loc, const char *&PrevSpec,
                       unsigned &DiagID, ParsedType Rep);
  bool SetTypeSpecType(TST T, SourceLocation Loc, const char *&PrevSpec,
                       unsigned &DiagID, Decl *Rep, bool Owned);
  bool SetTypeSpecType(TST T, SourceLocation TagKwLoc,
                       SourceLocation TagNameLoc, const char *&PrevSpec,
                       unsigned &DiagID, ParsedType Rep);
  bool SetTypeSpecType(TST T, SourceLocation TagKwLoc,
                       SourceLocation TagNameLoc, const char *&PrevSpec,
                       unsigned &DiagID, Decl *Rep, bool Owned);

  bool SetTypeSpecType(TST T, SourceLocation Loc, const char *&PrevSpec,
                       unsigned &DiagID, Expr *Rep);
  bool SetTypeAltiVecVector(bool isAltiVecVector, SourceLocation Loc,
                       const char *&PrevSpec, unsigned &DiagID);
  bool SetTypeAltiVecPixel(bool isAltiVecPixel, SourceLocation Loc,
                       const char *&PrevSpec, unsigned &DiagID);
  bool SetTypeSpecError();
  void UpdateDeclRep(Decl *Rep) {
    assert(isDeclRep((TST) TypeSpecType));
    DeclRep = Rep;
  }
  void UpdateTypeRep(ParsedType Rep) {
    assert(isTypeRep((TST) TypeSpecType));
    TypeRep = Rep;
  }
  void UpdateExprRep(Expr *Rep) {
    assert(isExprRep((TST) TypeSpecType));
    ExprRep = Rep;
  }

  bool SetTypeQual(TQ T, SourceLocation Loc, const char *&PrevSpec,
                   unsigned &DiagID, const LangOptions &Lang);

  bool SetFunctionSpecInline(SourceLocation Loc, const char *&PrevSpec,
                             unsigned &DiagID);
  bool SetFunctionSpecVirtual(SourceLocation Loc, const char *&PrevSpec,
                              unsigned &DiagID);
  bool SetFunctionSpecExplicit(SourceLocation Loc, const char *&PrevSpec,
                               unsigned &DiagID);

  bool SetFriendSpec(SourceLocation Loc, const char *&PrevSpec,
                     unsigned &DiagID);
  bool setModulePrivateSpec(SourceLocation Loc, const char *&PrevSpec,
                            unsigned &DiagID);
  bool SetConstexprSpec(SourceLocation Loc, const char *&PrevSpec,
                        unsigned &DiagID);

  bool isFriendSpecified() const { return Friend_specified; }
  SourceLocation getFriendSpecLoc() const { return FriendLoc; }

  bool isModulePrivateSpecified() const { return ModulePrivateLoc.isValid(); }
  SourceLocation getModulePrivateSpecLoc() const { return ModulePrivateLoc; }
  
  bool isConstexprSpecified() const { return Constexpr_specified; }
  SourceLocation getConstexprSpecLoc() const { return ConstexprLoc; }

  void ClearConstexprSpec() {
    Constexpr_specified = false;
    ConstexprLoc = SourceLocation();
  }

  AttributePool &getAttributePool() const {
    return Attrs.getPool();
  }

  /// \brief Concatenates two attribute lists.
  ///
  /// The GCC attribute syntax allows for the following:
  ///
  /// \code
  /// short __attribute__(( unused, deprecated ))
  /// int __attribute__(( may_alias, aligned(16) )) var;
  /// \endcode
  ///
  /// This declares 4 attributes using 2 lists. The following syntax is
  /// also allowed and equivalent to the previous declaration.
  ///
  /// \code
  /// short __attribute__((unused)) __attribute__((deprecated))
  /// int __attribute__((may_alias)) __attribute__((aligned(16))) var;
  /// \endcode
  ///
  void addAttributes(AttributeList *AL) {
    Attrs.addAll(AL);
  }
  void setAttributes(AttributeList *AL) {
    Attrs.set(AL);
  }

  bool hasAttributes() const { return !Attrs.empty(); }

  ParsedAttributes &getAttributes() { return Attrs; }
  const ParsedAttributes &getAttributes() const { return Attrs; }

  /// \brief Return the current attribute list and remove them from
  /// the DeclSpec so that it doesn't own them.
  ParsedAttributes takeAttributes() {
    // The non-const "copy" constructor clears the operand automatically.
    return Attrs;
  }

  void takeAttributesFrom(ParsedAttributes &attrs) {
    Attrs.takeAllFrom(attrs);
  }

  typedef Decl * const *ProtocolQualifierListTy;
  ProtocolQualifierListTy getProtocolQualifiers() const {
    return ProtocolQualifiers;
  }
  SourceLocation *getProtocolLocs() const { return ProtocolLocs; }
  unsigned getNumProtocolQualifiers() const {
    return NumProtocolQualifiers;
  }
  SourceLocation getProtocolLAngleLoc() const { return ProtocolLAngleLoc; }
  void setProtocolQualifiers(Decl * const *Protos, unsigned NP,
                             SourceLocation *ProtoLocs,
                             SourceLocation LAngleLoc);

  /// Finish - This does final analysis of the declspec, issuing diagnostics for
  /// things like "_Imaginary" (lacking an FP type).  After calling this method,
  /// DeclSpec is guaranteed self-consistent, even if an error occurred.
  void Finish(DiagnosticsEngine &D, Preprocessor &PP);

  const WrittenBuiltinSpecs& getWrittenBuiltinSpecs() const {
    return writtenBS;
  }

  ObjCDeclSpec *getObjCQualifiers() const { return ObjCQualifiers; }
  void setObjCQualifiers(ObjCDeclSpec *quals) { ObjCQualifiers = quals; }

  /// \brief Checks if this DeclSpec can stand alone, without a Declarator.
  ///
  /// Only tag declspecs can stand alone.
  bool isMissingDeclaratorOk();
};

/// \brief Captures information about "declaration specifiers" specific to
/// Objective-C.
class ObjCDeclSpec {
public:
  /// ObjCDeclQualifier - Qualifier used on types in method
  /// declarations.  Not all combinations are sensible.  Parameters
  /// can be one of { in, out, inout } with one of { bycopy, byref }.
  /// Returns can either be { oneway } or not.
  ///
  /// This should be kept in sync with Decl::ObjCDeclQualifier.
  enum ObjCDeclQualifier {
    DQ_None = 0x0,
    DQ_In = 0x1,
    DQ_Inout = 0x2,
    DQ_Out = 0x4,
    DQ_Bycopy = 0x8,
    DQ_Byref = 0x10,
    DQ_Oneway = 0x20
  };

  /// PropertyAttributeKind - list of property attributes.
  enum ObjCPropertyAttributeKind {
    DQ_PR_noattr = 0x0,
    DQ_PR_readonly = 0x01,
    DQ_PR_getter = 0x02,
    DQ_PR_assign = 0x04,
    DQ_PR_readwrite = 0x08,
    DQ_PR_retain = 0x10,
    DQ_PR_copy = 0x20,
    DQ_PR_nonatomic = 0x40,
    DQ_PR_setter = 0x80,
    DQ_PR_atomic = 0x100,
    DQ_PR_weak =   0x200,
    DQ_PR_strong = 0x400,
    DQ_PR_unsafe_unretained = 0x800
  };


  ObjCDeclSpec()
    : objcDeclQualifier(DQ_None), PropertyAttributes(DQ_PR_noattr),
      GetterName(0), SetterName(0) { }
  ObjCDeclQualifier getObjCDeclQualifier() const { return objcDeclQualifier; }
  void setObjCDeclQualifier(ObjCDeclQualifier DQVal) {
    objcDeclQualifier = (ObjCDeclQualifier) (objcDeclQualifier | DQVal);
  }

  ObjCPropertyAttributeKind getPropertyAttributes() const {
    return ObjCPropertyAttributeKind(PropertyAttributes);
  }
  void setPropertyAttributes(ObjCPropertyAttributeKind PRVal) {
    PropertyAttributes =
      (ObjCPropertyAttributeKind)(PropertyAttributes | PRVal);
  }

  const IdentifierInfo *getGetterName() const { return GetterName; }
  IdentifierInfo *getGetterName() { return GetterName; }
  void setGetterName(IdentifierInfo *name) { GetterName = name; }

  const IdentifierInfo *getSetterName() const { return SetterName; }
  IdentifierInfo *getSetterName() { return SetterName; }
  void setSetterName(IdentifierInfo *name) { SetterName = name; }

private:
  // FIXME: These two are unrelated and mutially exclusive. So perhaps
  // we can put them in a union to reflect their mutual exclusiveness
  // (space saving is negligible).
  ObjCDeclQualifier objcDeclQualifier : 6;

  // NOTE: VC++ treats enums as signed, avoid using ObjCPropertyAttributeKind
  unsigned PropertyAttributes : 12;
  IdentifierInfo *GetterName;    // getter name of NULL if no getter
  IdentifierInfo *SetterName;    // setter name of NULL if no setter
};

/// \brief Represents a C++ unqualified-id that has been parsed. 
class UnqualifiedId {
private:
  UnqualifiedId(const UnqualifiedId &Other) LLVM_DELETED_FUNCTION;
  const UnqualifiedId &operator=(const UnqualifiedId &) LLVM_DELETED_FUNCTION;

public:
  /// \brief Describes the kind of unqualified-id parsed.
  enum IdKind {
    /// \brief An identifier.
    IK_Identifier,
    /// \brief An overloaded operator name, e.g., operator+.
    IK_OperatorFunctionId,
    /// \brief A conversion function name, e.g., operator int.
    IK_ConversionFunctionId,
    /// \brief A user-defined literal name, e.g., operator "" _i.
    IK_LiteralOperatorId,
    /// \brief A constructor name.
    IK_ConstructorName,
    /// \brief A constructor named via a template-id.
    IK_ConstructorTemplateId,
    /// \brief A destructor name.
    IK_DestructorName,
    /// \brief A template-id, e.g., f<int>.
    IK_TemplateId,
    /// \brief An implicit 'self' parameter
    IK_ImplicitSelfParam
  } Kind;

  /// \brief Anonymous union that holds extra data associated with the
  /// parsed unqualified-id.
  union {
    /// \brief When Kind == IK_Identifier, the parsed identifier, or when Kind
    /// == IK_UserLiteralId, the identifier suffix.
    IdentifierInfo *Identifier;
    
    /// \brief When Kind == IK_OperatorFunctionId, the overloaded operator
    /// that we parsed.
    struct {
      /// \brief The kind of overloaded operator.
      OverloadedOperatorKind Operator;
      
      /// \brief The source locations of the individual tokens that name
      /// the operator, e.g., the "new", "[", and "]" tokens in 
      /// operator new []. 
      ///
      /// Different operators have different numbers of tokens in their name,
      /// up to three. Any remaining source locations in this array will be
      /// set to an invalid value for operators with fewer than three tokens.
      unsigned SymbolLocations[3];
    } OperatorFunctionId;
    
    /// \brief When Kind == IK_ConversionFunctionId, the type that the 
    /// conversion function names.
    UnionParsedType ConversionFunctionId;

    /// \brief When Kind == IK_ConstructorName, the class-name of the type
    /// whose constructor is being referenced.
    UnionParsedType ConstructorName;
    
    /// \brief When Kind == IK_DestructorName, the type referred to by the
    /// class-name.
    UnionParsedType DestructorName;
    
    /// \brief When Kind == IK_TemplateId or IK_ConstructorTemplateId,
    /// the template-id annotation that contains the template name and
    /// template arguments.
    TemplateIdAnnotation *TemplateId;
  };
  
  /// \brief The location of the first token that describes this unqualified-id,
  /// which will be the location of the identifier, "operator" keyword,
  /// tilde (for a destructor), or the template name of a template-id.
  SourceLocation StartLocation;
  
  /// \brief The location of the last token that describes this unqualified-id.
  SourceLocation EndLocation;
  
  UnqualifiedId() : Kind(IK_Identifier), Identifier(0) { }

  /// \brief Clear out this unqualified-id, setting it to default (invalid) 
  /// state.
  void clear() {
    Kind = IK_Identifier;
    Identifier = 0;
    StartLocation = SourceLocation();
    EndLocation = SourceLocation();
  }
  
  /// \brief Determine whether this unqualified-id refers to a valid name.
  bool isValid() const { return StartLocation.isValid(); }

  /// \brief Determine whether this unqualified-id refers to an invalid name.
  bool isInvalid() const { return !isValid(); }
  
  /// \brief Determine what kind of name we have.
  IdKind getKind() const { return Kind; }
  void setKind(IdKind kind) { Kind = kind; } 
  
  /// \brief Specify that this unqualified-id was parsed as an identifier.
  ///
  /// \param Id the parsed identifier.
  /// \param IdLoc the location of the parsed identifier.
  void setIdentifier(const IdentifierInfo *Id, SourceLocation IdLoc) {
    Kind = IK_Identifier;
    Identifier = const_cast<IdentifierInfo *>(Id);
    StartLocation = EndLocation = IdLoc;
  }
  
  /// \brief Specify that this unqualified-id was parsed as an 
  /// operator-function-id.
  ///
  /// \param OperatorLoc the location of the 'operator' keyword.
  ///
  /// \param Op the overloaded operator.
  ///
  /// \param SymbolLocations the locations of the individual operator symbols
  /// in the operator.
  void setOperatorFunctionId(SourceLocation OperatorLoc, 
                             OverloadedOperatorKind Op,
                             SourceLocation SymbolLocations[3]);
  
  /// \brief Specify that this unqualified-id was parsed as a 
  /// conversion-function-id.
  ///
  /// \param OperatorLoc the location of the 'operator' keyword.
  ///
  /// \param Ty the type to which this conversion function is converting.
  ///
  /// \param EndLoc the location of the last token that makes up the type name.
  void setConversionFunctionId(SourceLocation OperatorLoc, 
                               ParsedType Ty,
                               SourceLocation EndLoc) {
    Kind = IK_ConversionFunctionId;
    StartLocation = OperatorLoc;
    EndLocation = EndLoc;
    ConversionFunctionId = Ty;
  }

  /// \brief Specific that this unqualified-id was parsed as a
  /// literal-operator-id.
  ///
  /// \param Id the parsed identifier.
  ///
  /// \param OpLoc the location of the 'operator' keyword.
  ///
  /// \param IdLoc the location of the identifier.
  void setLiteralOperatorId(const IdentifierInfo *Id, SourceLocation OpLoc,
                              SourceLocation IdLoc) {
    Kind = IK_LiteralOperatorId;
    Identifier = const_cast<IdentifierInfo *>(Id);
    StartLocation = OpLoc;
    EndLocation = IdLoc;
  }
  
  /// \brief Specify that this unqualified-id was parsed as a constructor name.
  ///
  /// \param ClassType the class type referred to by the constructor name.
  ///
  /// \param ClassNameLoc the location of the class name.
  ///
  /// \param EndLoc the location of the last token that makes up the type name.
  void setConstructorName(ParsedType ClassType, 
                          SourceLocation ClassNameLoc,
                          SourceLocation EndLoc) {
    Kind = IK_ConstructorName;
    StartLocation = ClassNameLoc;
    EndLocation = EndLoc;
    ConstructorName = ClassType;
  }

  /// \brief Specify that this unqualified-id was parsed as a
  /// template-id that names a constructor.
  ///
  /// \param TemplateId the template-id annotation that describes the parsed
  /// template-id. This UnqualifiedId instance will take ownership of the
  /// \p TemplateId and will free it on destruction.
  void setConstructorTemplateId(TemplateIdAnnotation *TemplateId);

  /// \brief Specify that this unqualified-id was parsed as a destructor name.
  ///
  /// \param TildeLoc the location of the '~' that introduces the destructor
  /// name.
  ///
  /// \param ClassType the name of the class referred to by the destructor name.
  void setDestructorName(SourceLocation TildeLoc,
                         ParsedType ClassType,
                         SourceLocation EndLoc) {
    Kind = IK_DestructorName;
    StartLocation = TildeLoc;
    EndLocation = EndLoc;
    DestructorName = ClassType;
  }
  
  /// \brief Specify that this unqualified-id was parsed as a template-id.
  ///
  /// \param TemplateId the template-id annotation that describes the parsed
  /// template-id. This UnqualifiedId instance will take ownership of the
  /// \p TemplateId and will free it on destruction.
  void setTemplateId(TemplateIdAnnotation *TemplateId);

  /// \brief Return the source range that covers this unqualified-id.
  SourceRange getSourceRange() const LLVM_READONLY { 
    return SourceRange(StartLocation, EndLocation); 
  }
  SourceLocation getLocStart() const LLVM_READONLY { return StartLocation; }
  SourceLocation getLocEnd() const LLVM_READONLY { return EndLocation; }
};

/// \brief A set of tokens that has been cached for later parsing.
typedef SmallVector<Token, 4> CachedTokens;

/// \brief One instance of this struct is used for each type in a
/// declarator that is parsed.
///
/// This is intended to be a small value object.
struct DeclaratorChunk {
  enum {
    Pointer, Reference, Array, Function, BlockPointer, MemberPointer, Paren
  } Kind;

  /// Loc - The place where this type was defined.
  SourceLocation Loc;
  /// EndLoc - If valid, the place where this chunck ends.
  SourceLocation EndLoc;

  struct TypeInfoCommon {
    AttributeList *AttrList;
  };

  struct PointerTypeInfo : TypeInfoCommon {
    /// The type qualifiers: const/volatile/restrict.
    unsigned TypeQuals : 3;

    /// The location of the const-qualifier, if any.
    unsigned ConstQualLoc;

    /// The location of the volatile-qualifier, if any.
    unsigned VolatileQualLoc;

    /// The location of the restrict-qualifier, if any.
    unsigned RestrictQualLoc;

    void destroy() {
    }
  };

  struct ReferenceTypeInfo : TypeInfoCommon {
    /// The type qualifier: restrict. [GNU] C++ extension
    bool HasRestrict : 1;
    /// True if this is an lvalue reference, false if it's an rvalue reference.
    bool LValueRef : 1;
    void destroy() {
    }
  };

  struct ArrayTypeInfo : TypeInfoCommon {
    /// The type qualifiers for the array: const/volatile/restrict.
    unsigned TypeQuals : 3;

    /// True if this dimension included the 'static' keyword.
    bool hasStatic : 1;

    /// True if this dimension was [*].  In this case, NumElts is null.
    bool isStar : 1;

    /// This is the size of the array, or null if [] or [*] was specified.
    /// Since the parser is multi-purpose, and we don't want to impose a root
    /// expression class on all clients, NumElts is untyped.
    Expr *NumElts;

    void destroy() {}
  };

  /// ParamInfo - An array of paraminfo objects is allocated whenever a function
  /// declarator is parsed.  There are two interesting styles of arguments here:
  /// K&R-style identifier lists and parameter type lists.  K&R-style identifier
  /// lists will have information about the identifier, but no type information.
  /// Parameter type lists will have type info (if the actions module provides
  /// it), but may have null identifier info: e.g. for 'void foo(int X, int)'.
  struct ParamInfo {
    IdentifierInfo *Ident;
    SourceLocation IdentLoc;
    Decl *Param;

    /// DefaultArgTokens - When the parameter's default argument
    /// cannot be parsed immediately (because it occurs within the
    /// declaration of a member function), it will be stored here as a
    /// sequence of tokens to be parsed once the class definition is
    /// complete. Non-NULL indicates that there is a default argument.
    CachedTokens *DefaultArgTokens;

    ParamInfo() {}
    ParamInfo(IdentifierInfo *ident, SourceLocation iloc,
              Decl *param,
              CachedTokens *DefArgTokens = 0)
      : Ident(ident), IdentLoc(iloc), Param(param),
        DefaultArgTokens(DefArgTokens) {}
  };

  struct TypeAndRange {
    ParsedType Ty;
    SourceRange Range;
  };

  struct FunctionTypeInfo : TypeInfoCommon {
    /// hasPrototype - This is true if the function had at least one typed
    /// argument.  If the function is () or (a,b,c), then it has no prototype,
    /// and is treated as a K&R-style function.
    unsigned hasPrototype : 1;

    /// isVariadic - If this function has a prototype, and if that
    /// proto ends with ',...)', this is true. When true, EllipsisLoc
    /// contains the location of the ellipsis.
    unsigned isVariadic : 1;

    /// Can this declaration be a constructor-style initializer?
    unsigned isAmbiguous : 1;

    /// \brief Whether the ref-qualifier (if any) is an lvalue reference.
    /// Otherwise, it's an rvalue reference.
    unsigned RefQualifierIsLValueRef : 1;

    /// The type qualifiers: const/volatile/restrict.
    /// The qualifier bitmask values are the same as in QualType.
    unsigned TypeQuals : 3;

    /// ExceptionSpecType - An ExceptionSpecificationType value.
    unsigned ExceptionSpecType : 3;

    /// DeleteArgInfo - If this is true, we need to delete[] ArgInfo.
    unsigned DeleteArgInfo : 1;

    /// HasTrailingReturnType - If this is true, a trailing return type was
    /// specified.
    unsigned HasTrailingReturnType : 1;

    /// The location of the left parenthesis in the source.
    unsigned LParenLoc;

    /// When isVariadic is true, the location of the ellipsis in the source.
    unsigned EllipsisLoc;

    /// The location of the right parenthesis in the source.
    unsigned RParenLoc;

    /// NumArgs - This is the number of formal arguments provided for the
    /// declarator.
    unsigned NumArgs;

    /// NumExceptions - This is the number of types in the dynamic-exception-
    /// decl, if the function has one.
    unsigned NumExceptions;

    /// \brief The location of the ref-qualifier, if any.
    ///
    /// If this is an invalid location, there is no ref-qualifier.
    unsigned RefQualifierLoc;

    /// \brief The location of the const-qualifier, if any.
    ///
    /// If this is an invalid location, there is no const-qualifier.
    unsigned ConstQualifierLoc;

    /// \brief The location of the volatile-qualifier, if any.
    ///
    /// If this is an invalid location, there is no volatile-qualifier.
    unsigned VolatileQualifierLoc;

    /// \brief The location of the 'mutable' qualifer in a lambda-declarator, if
    /// any.
    unsigned MutableLoc;

    /// \brief The location of the keyword introducing the spec, if any.
    unsigned ExceptionSpecLoc;

    /// ArgInfo - This is a pointer to a new[]'d array of ParamInfo objects that
    /// describe the arguments for this function declarator.  This is null if
    /// there are no arguments specified.
    ParamInfo *ArgInfo;

    union {
      /// \brief Pointer to a new[]'d array of TypeAndRange objects that
      /// contain the types in the function's dynamic exception specification
      /// and their locations, if there is one.
      TypeAndRange *Exceptions;

      /// \brief Pointer to the expression in the noexcept-specifier of this
      /// function, if it has one.
      Expr *NoexceptExpr;
    };

    /// \brief If HasTrailingReturnType is true, this is the trailing return
    /// type specified.
    UnionParsedType TrailingReturnType;

    /// \brief Reset the argument list to having zero arguments.
    ///
    /// This is used in various places for error recovery.
    void freeArgs() {
      if (DeleteArgInfo) {
        delete[] ArgInfo;
        DeleteArgInfo = false;
      }
      NumArgs = 0;
    }

    void destroy() {
      if (DeleteArgInfo)
        delete[] ArgInfo;
      if (getExceptionSpecType() == EST_Dynamic)
        delete[] Exceptions;
    }

    /// isKNRPrototype - Return true if this is a K&R style identifier list,
    /// like "void foo(a,b,c)".  In a function definition, this will be followed
    /// by the argument type definitions.
    bool isKNRPrototype() const {
      return !hasPrototype && NumArgs != 0;
    }

    SourceLocation getLParenLoc() const {
      return SourceLocation::getFromRawEncoding(LParenLoc);
    }

    SourceLocation getEllipsisLoc() const {
      return SourceLocation::getFromRawEncoding(EllipsisLoc);
    }

    SourceLocation getRParenLoc() const {
      return SourceLocation::getFromRawEncoding(RParenLoc);
    }

    SourceLocation getExceptionSpecLoc() const {
      return SourceLocation::getFromRawEncoding(ExceptionSpecLoc);
    }

    /// \brief Retrieve the location of the ref-qualifier, if any.
    SourceLocation getRefQualifierLoc() const {
      return SourceLocation::getFromRawEncoding(RefQualifierLoc);
    }

    /// \brief Retrieve the location of the ref-qualifier, if any.
    SourceLocation getConstQualifierLoc() const {
      return SourceLocation::getFromRawEncoding(ConstQualifierLoc);
    }

    /// \brief Retrieve the location of the ref-qualifier, if any.
    SourceLocation getVolatileQualifierLoc() const {
      return SourceLocation::getFromRawEncoding(VolatileQualifierLoc);
    }

    /// \brief Retrieve the location of the 'mutable' qualifier, if any.
    SourceLocation getMutableLoc() const {
      return SourceLocation::getFromRawEncoding(MutableLoc);
    }

    /// \brief Determine whether this function declaration contains a 
    /// ref-qualifier.
    bool hasRefQualifier() const { return getRefQualifierLoc().isValid(); }

    /// \brief Determine whether this lambda-declarator contains a 'mutable'
    /// qualifier.
    bool hasMutableQualifier() const { return getMutableLoc().isValid(); }

    /// \brief Get the type of exception specification this function has.
    ExceptionSpecificationType getExceptionSpecType() const {
      return static_cast<ExceptionSpecificationType>(ExceptionSpecType);
    }

    /// \brief Determine whether this function declarator had a
    /// trailing-return-type.
    bool hasTrailingReturnType() const { return HasTrailingReturnType; }

    /// \brief Get the trailing-return-type for this function declarator.
    ParsedType getTrailingReturnType() const { return TrailingReturnType; }
  };

  struct BlockPointerTypeInfo : TypeInfoCommon {
    /// For now, sema will catch these as invalid.
    /// The type qualifiers: const/volatile/restrict.
    unsigned TypeQuals : 3;

    void destroy() {
    }
  };

  struct MemberPointerTypeInfo : TypeInfoCommon {
    /// The type qualifiers: const/volatile/restrict.
    unsigned TypeQuals : 3;
    // CXXScopeSpec has a constructor, so it can't be a direct member.
    // So we need some pointer-aligned storage and a bit of trickery.
    union {
      void *Aligner;
      char Mem[sizeof(CXXScopeSpec)];
    } ScopeMem;
    CXXScopeSpec &Scope() {
      return *reinterpret_cast<CXXScopeSpec*>(ScopeMem.Mem);
    }
    const CXXScopeSpec &Scope() const {
      return *reinterpret_cast<const CXXScopeSpec*>(ScopeMem.Mem);
    }
    void destroy() {
      Scope().~CXXScopeSpec();
    }
  };

  union {
    TypeInfoCommon        Common;
    PointerTypeInfo       Ptr;
    ReferenceTypeInfo     Ref;
    ArrayTypeInfo         Arr;
    FunctionTypeInfo      Fun;
    BlockPointerTypeInfo  Cls;
    MemberPointerTypeInfo Mem;
  };

  void destroy() {
    switch (Kind) {
    case DeclaratorChunk::Function:      return Fun.destroy();
    case DeclaratorChunk::Pointer:       return Ptr.destroy();
    case DeclaratorChunk::BlockPointer:  return Cls.destroy();
    case DeclaratorChunk::Reference:     return Ref.destroy();
    case DeclaratorChunk::Array:         return Arr.destroy();
    case DeclaratorChunk::MemberPointer: return Mem.destroy();
    case DeclaratorChunk::Paren:         return;
    }
  }

  /// \brief If there are attributes applied to this declaratorchunk, return
  /// them.
  const AttributeList *getAttrs() const {
    return Common.AttrList;
  }

  AttributeList *&getAttrListRef() {
    return Common.AttrList;
  }

  /// \brief Return a DeclaratorChunk for a pointer.
  static DeclaratorChunk getPointer(unsigned TypeQuals, SourceLocation Loc,
                                    SourceLocation ConstQualLoc,
                                    SourceLocation VolatileQualLoc,
                                    SourceLocation RestrictQualLoc) {
    DeclaratorChunk I;
    I.Kind                = Pointer;
    I.Loc                 = Loc;
    I.Ptr.TypeQuals       = TypeQuals;
    I.Ptr.ConstQualLoc    = ConstQualLoc.getRawEncoding();
    I.Ptr.VolatileQualLoc = VolatileQualLoc.getRawEncoding();
    I.Ptr.RestrictQualLoc = RestrictQualLoc.getRawEncoding();
    I.Ptr.AttrList        = 0;
    return I;
  }

  /// \brief Return a DeclaratorChunk for a reference.
  static DeclaratorChunk getReference(unsigned TypeQuals, SourceLocation Loc,
                                      bool lvalue) {
    DeclaratorChunk I;
    I.Kind            = Reference;
    I.Loc             = Loc;
    I.Ref.HasRestrict = (TypeQuals & DeclSpec::TQ_restrict) != 0;
    I.Ref.LValueRef   = lvalue;
    I.Ref.AttrList    = 0;
    return I;
  }

  /// \brief Return a DeclaratorChunk for an array.
  static DeclaratorChunk getArray(unsigned TypeQuals,
                                  bool isStatic, bool isStar, Expr *NumElts,
                                  SourceLocation LBLoc, SourceLocation RBLoc) {
    DeclaratorChunk I;
    I.Kind          = Array;
    I.Loc           = LBLoc;
    I.EndLoc        = RBLoc;
    I.Arr.AttrList  = 0;
    I.Arr.TypeQuals = TypeQuals;
    I.Arr.hasStatic = isStatic;
    I.Arr.isStar    = isStar;
    I.Arr.NumElts   = NumElts;
    return I;
  }

  /// DeclaratorChunk::getFunction - Return a DeclaratorChunk for a function.
  /// "TheDeclarator" is the declarator that this will be added to.
  static DeclaratorChunk getFunction(bool hasProto,
                                     bool isAmbiguous,
                                     SourceLocation LParenLoc,
                                     ParamInfo *ArgInfo, unsigned NumArgs,
                                     SourceLocation EllipsisLoc,
                                     SourceLocation RParenLoc,
                                     unsigned TypeQuals,
                                     bool RefQualifierIsLvalueRef,
                                     SourceLocation RefQualifierLoc,
                                     SourceLocation ConstQualifierLoc,
                                     SourceLocation VolatileQualifierLoc,
                                     SourceLocation MutableLoc,
                                     ExceptionSpecificationType ESpecType,
                                     SourceLocation ESpecLoc,
                                     ParsedType *Exceptions,
                                     SourceRange *ExceptionRanges,
                                     unsigned NumExceptions,
                                     Expr *NoexceptExpr,
                                     SourceLocation LocalRangeBegin,
                                     SourceLocation LocalRangeEnd,
                                     Declarator &TheDeclarator,
                                     TypeResult TrailingReturnType =
                                                    TypeResult());

  /// \brief Return a DeclaratorChunk for a block.
  static DeclaratorChunk getBlockPointer(unsigned TypeQuals,
                                         SourceLocation Loc) {
    DeclaratorChunk I;
    I.Kind          = BlockPointer;
    I.Loc           = Loc;
    I.Cls.TypeQuals = TypeQuals;
    I.Cls.AttrList  = 0;
    return I;
  }

  static DeclaratorChunk getMemberPointer(const CXXScopeSpec &SS,
                                          unsigned TypeQuals,
                                          SourceLocation Loc) {
    DeclaratorChunk I;
    I.Kind          = MemberPointer;
    I.Loc           = Loc;
    I.Mem.TypeQuals = TypeQuals;
    I.Mem.AttrList  = 0;
    new (I.Mem.ScopeMem.Mem) CXXScopeSpec(SS);
    return I;
  }

  /// \brief Return a DeclaratorChunk for a paren.
  static DeclaratorChunk getParen(SourceLocation LParenLoc,
                                  SourceLocation RParenLoc) {
    DeclaratorChunk I;
    I.Kind          = Paren;
    I.Loc           = LParenLoc;
    I.EndLoc        = RParenLoc;
    I.Common.AttrList = 0;
    return I;
  }

};

/// \brief Described the kind of function definition (if any) provided for
/// a function.
enum FunctionDefinitionKind {
  FDK_Declaration,
  FDK_Definition,
  FDK_Defaulted,
  FDK_Deleted
};

/// \brief Information about one declarator, including the parsed type
/// information and the identifier.
///
/// When the declarator is fully formed, this is turned into the appropriate
/// Decl object.
///
/// Declarators come in two types: normal declarators and abstract declarators.
/// Abstract declarators are used when parsing types, and don't have an
/// identifier.  Normal declarators do have ID's.
///
/// Instances of this class should be a transient object that lives on the
/// stack, not objects that are allocated in large quantities on the heap.
class Declarator {
public:
  enum TheContext {
    FileContext,         // File scope declaration.
    PrototypeContext,    // Within a function prototype.
    ObjCResultContext,   // An ObjC method result type.
    ObjCParameterContext,// An ObjC method parameter type.
    KNRTypeListContext,  // K&R type definition list for formals.
    TypeNameContext,     // Abstract declarator for types.
    MemberContext,       // Struct/Union field.
    BlockContext,        // Declaration within a block in a function.
    ForContext,          // Declaration within first part of a for loop.
    ConditionContext,    // Condition declaration in a C++ if/switch/while/for.
    TemplateParamContext,// Within a template parameter list.
    CXXNewContext,       // C++ new-expression.
    CXXCatchContext,     // C++ catch exception-declaration
    ObjCCatchContext,    // Objective-C catch exception-declaration
    BlockLiteralContext,  // Block literal declarator.
    LambdaExprContext,   // Lambda-expression declarator.
    TrailingReturnContext, // C++11 trailing-type-specifier.
    TemplateTypeArgContext, // Template type argument.
    AliasDeclContext,    // C++11 alias-declaration.
    AliasTemplateContext // C++11 alias-declaration template.
  };

private:
  const DeclSpec &DS;
  CXXScopeSpec SS;
  UnqualifiedId Name;
  SourceRange Range;

  /// \brief Where we are parsing this declarator.
  TheContext Context;

  /// DeclTypeInfo - This holds each type that the declarator includes as it is
  /// parsed.  This is pushed from the identifier out, which means that element
  /// #0 will be the most closely bound to the identifier, and
  /// DeclTypeInfo.back() will be the least closely bound.
  SmallVector<DeclaratorChunk, 8> DeclTypeInfo;

  /// InvalidType - Set by Sema::GetTypeForDeclarator().
  bool InvalidType : 1;

  /// GroupingParens - Set by Parser::ParseParenDeclarator().
  bool GroupingParens : 1;

  /// FunctionDefinition - Is this Declarator for a function or member 
  /// definition and, if so, what kind?
  ///
  /// Actually a FunctionDefinitionKind.
  unsigned FunctionDefinition : 2;

  /// \brief Is this Declarator a redeclaration?
  bool Redeclaration : 1;

  /// Attrs - Attributes.
  ParsedAttributes Attrs;

  /// \brief The asm label, if specified.
  Expr *AsmLabel;

  /// InlineParams - This is a local array used for the first function decl
  /// chunk to avoid going to the heap for the common case when we have one
  /// function chunk in the declarator.
  DeclaratorChunk::ParamInfo InlineParams[16];
  bool InlineParamsUsed;

  /// \brief true if the declaration is preceded by \c __extension__.
  bool Extension : 1;

  /// \brief If this is the second or subsequent declarator in this declaration,
  /// the location of the comma before this declarator.
  SourceLocation CommaLoc;

  /// \brief If provided, the source location of the ellipsis used to describe
  /// this declarator as a parameter pack.
  SourceLocation EllipsisLoc;
  
  friend struct DeclaratorChunk;

public:
  Declarator(const DeclSpec &ds, TheContext C)
    : DS(ds), Range(ds.getSourceRange()), Context(C),
      InvalidType(DS.getTypeSpecType() == DeclSpec::TST_error),
      GroupingParens(false), FunctionDefinition(FDK_Declaration), 
      Redeclaration(false),
      Attrs(ds.getAttributePool().getFactory()), AsmLabel(0),
      InlineParamsUsed(false), Extension(false) {
  }

  ~Declarator() {
    clear();
  }

  /// getDeclSpec - Return the declaration-specifier that this declarator was
  /// declared with.
  const DeclSpec &getDeclSpec() const { return DS; }

  /// getMutableDeclSpec - Return a non-const version of the DeclSpec.  This
  /// should be used with extreme care: declspecs can often be shared between
  /// multiple declarators, so mutating the DeclSpec affects all of the
  /// Declarators.  This should only be done when the declspec is known to not
  /// be shared or when in error recovery etc.
  DeclSpec &getMutableDeclSpec() { return const_cast<DeclSpec &>(DS); }

  AttributePool &getAttributePool() const {
    return Attrs.getPool();
  }

  /// getCXXScopeSpec - Return the C++ scope specifier (global scope or
  /// nested-name-specifier) that is part of the declarator-id.
  const CXXScopeSpec &getCXXScopeSpec() const { return SS; }
  CXXScopeSpec &getCXXScopeSpec() { return SS; }

  /// \brief Retrieve the name specified by this declarator.
  UnqualifiedId &getName() { return Name; }
  
  TheContext getContext() const { return Context; }

  bool isPrototypeContext() const {
    return (Context == PrototypeContext ||
            Context == ObjCParameterContext ||
            Context == ObjCResultContext);
  }

  /// \brief Get the source range that spans this declarator.
  const SourceRange &getSourceRange() const LLVM_READONLY { return Range; }
  SourceLocation getLocStart() const LLVM_READONLY { return Range.getBegin(); }
  SourceLocation getLocEnd() const LLVM_READONLY { return Range.getEnd(); }

  void SetSourceRange(SourceRange R) { Range = R; }
  /// SetRangeBegin - Set the start of the source range to Loc, unless it's
  /// invalid.
  void SetRangeBegin(SourceLocation Loc) {
    if (!Loc.isInvalid())
      Range.setBegin(Loc);
  }
  /// SetRangeEnd - Set the end of the source range to Loc, unless it's invalid.
  void SetRangeEnd(SourceLocation Loc) {
    if (!Loc.isInvalid())
      Range.setEnd(Loc);
  }
  /// ExtendWithDeclSpec - Extend the declarator source range to include the
  /// given declspec, unless its location is invalid. Adopts the range start if
  /// the current range start is invalid.
  void ExtendWithDeclSpec(const DeclSpec &DS) {
    const SourceRange &SR = DS.getSourceRange();
    if (Range.getBegin().isInvalid())
      Range.setBegin(SR.getBegin());
    if (!SR.getEnd().isInvalid())
      Range.setEnd(SR.getEnd());
  }

  /// \brief Reset the contents of this Declarator.
  void clear() {
    SS.clear();
    Name.clear();
    Range = DS.getSourceRange();
    
    for (unsigned i = 0, e = DeclTypeInfo.size(); i != e; ++i)
      DeclTypeInfo[i].destroy();
    DeclTypeInfo.clear();
    Attrs.clear();
    AsmLabel = 0;
    InlineParamsUsed = false;
    CommaLoc = SourceLocation();
    EllipsisLoc = SourceLocation();
  }

  /// mayOmitIdentifier - Return true if the identifier is either optional or
  /// not allowed.  This is true for typenames, prototypes, and template
  /// parameter lists.
  bool mayOmitIdentifier() const {
    switch (Context) {
    case FileContext:
    case KNRTypeListContext:
    case MemberContext:
    case BlockContext:
    case ForContext:
    case ConditionContext:
      return false;

    case TypeNameContext:
    case AliasDeclContext:
    case AliasTemplateContext:
    case PrototypeContext:
    case ObjCParameterContext:
    case ObjCResultContext:
    case TemplateParamContext:
    case CXXNewContext:
    case CXXCatchContext:
    case ObjCCatchContext:
    case BlockLiteralContext:
    case LambdaExprContext:
    case TemplateTypeArgContext:
    case TrailingReturnContext:
      return true;
    }
    llvm_unreachable("unknown context kind!");
  }

  /// mayHaveIdentifier - Return true if the identifier is either optional or
  /// required.  This is true for normal declarators and prototypes, but not
  /// typenames.
  bool mayHaveIdentifier() const {
    switch (Context) {
    case FileContext:
    case KNRTypeListContext:
    case MemberContext:
    case BlockContext:
    case ForContext:
    case ConditionContext:
    case PrototypeContext:
    case TemplateParamContext:
    case CXXCatchContext:
    case ObjCCatchContext:
      return true;

    case TypeNameContext:
    case CXXNewContext:
    case AliasDeclContext:
    case AliasTemplateContext:
    case ObjCParameterContext:
    case ObjCResultContext:
    case BlockLiteralContext:
    case LambdaExprContext:
    case TemplateTypeArgContext:
    case TrailingReturnContext:
      return false;
    }
    llvm_unreachable("unknown context kind!");
  }

  /// mayBeFollowedByCXXDirectInit - Return true if the declarator can be
  /// followed by a C++ direct initializer, e.g. "int x(1);".
  bool mayBeFollowedByCXXDirectInit() const {
    if (hasGroupingParens()) return false;

    if (getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_typedef)
      return false;

    if (getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_extern &&
        Context != FileContext)
      return false;

    // Special names can't have direct initializers.
    if (Name.getKind() != UnqualifiedId::IK_Identifier)
      return false;

    switch (Context) {
    case FileContext:
    case BlockContext:
    case ForContext:
      return true;

    case ConditionContext:
      // This may not be followed by a direct initializer, but it can't be a
      // function declaration either, and we'd prefer to perform a tentative
      // parse in order to produce the right diagnostic.
      return true;

    case KNRTypeListContext:
    case MemberContext:
    case PrototypeContext:
    case ObjCParameterContext:
    case ObjCResultContext:
    case TemplateParamContext:
    case CXXCatchContext:
    case ObjCCatchContext:
    case TypeNameContext:
    case CXXNewContext:
    case AliasDeclContext:
    case AliasTemplateContext:
    case BlockLiteralContext:
    case LambdaExprContext:
    case TemplateTypeArgContext:
    case TrailingReturnContext:
      return false;
    }
    llvm_unreachable("unknown context kind!");
  }

  /// isPastIdentifier - Return true if we have parsed beyond the point where
  /// the
  bool isPastIdentifier() const { return Name.isValid(); }

  /// hasName - Whether this declarator has a name, which might be an
  /// identifier (accessible via getIdentifier()) or some kind of
  /// special C++ name (constructor, destructor, etc.).
  bool hasName() const { 
    return Name.getKind() != UnqualifiedId::IK_Identifier || Name.Identifier;
  }

  IdentifierInfo *getIdentifier() const { 
    if (Name.getKind() == UnqualifiedId::IK_Identifier)
      return Name.Identifier;
    
    return 0;
  }
  SourceLocation getIdentifierLoc() const { return Name.StartLocation; }

  /// \brief Set the name of this declarator to be the given identifier.
  void SetIdentifier(IdentifierInfo *Id, SourceLocation IdLoc) {
    Name.setIdentifier(Id, IdLoc);
  }
  
  /// AddTypeInfo - Add a chunk to this declarator. Also extend the range to
  /// EndLoc, which should be the last token of the chunk.
  void AddTypeInfo(const DeclaratorChunk &TI,
                   ParsedAttributes &attrs,
                   SourceLocation EndLoc) {
    DeclTypeInfo.push_back(TI);
    DeclTypeInfo.back().getAttrListRef() = attrs.getList();
    getAttributePool().takeAllFrom(attrs.getPool());

    if (!EndLoc.isInvalid())
      SetRangeEnd(EndLoc);
  }

  /// \brief Add a new innermost chunk to this declarator.
  void AddInnermostTypeInfo(const DeclaratorChunk &TI) {
    DeclTypeInfo.insert(DeclTypeInfo.begin(), TI);
  }

  /// \brief Return the number of types applied to this declarator.
  unsigned getNumTypeObjects() const { return DeclTypeInfo.size(); }

  /// Return the specified TypeInfo from this declarator.  TypeInfo #0 is
  /// closest to the identifier.
  const DeclaratorChunk &getTypeObject(unsigned i) const {
    assert(i < DeclTypeInfo.size() && "Invalid type chunk");
    return DeclTypeInfo[i];
  }
  DeclaratorChunk &getTypeObject(unsigned i) {
    assert(i < DeclTypeInfo.size() && "Invalid type chunk");
    return DeclTypeInfo[i];
  }

  void DropFirstTypeObject()
  {
    assert(!DeclTypeInfo.empty() && "No type chunks to drop.");
    DeclTypeInfo.front().destroy();
    DeclTypeInfo.erase(DeclTypeInfo.begin());
  }

  /// isArrayOfUnknownBound - This method returns true if the declarator
  /// is a declarator for an array of unknown bound (looking through
  /// parentheses).
  bool isArrayOfUnknownBound() const {
    for (unsigned i = 0, i_end = DeclTypeInfo.size(); i < i_end; ++i) {
      switch (DeclTypeInfo[i].Kind) {
      case DeclaratorChunk::Paren:
        continue;
      case DeclaratorChunk::Function:
      case DeclaratorChunk::Pointer:
      case DeclaratorChunk::Reference:
      case DeclaratorChunk::BlockPointer:
      case DeclaratorChunk::MemberPointer:
        return false;
      case DeclaratorChunk::Array:
        return !DeclTypeInfo[i].Arr.NumElts;
      }
      llvm_unreachable("Invalid type chunk");
    }
    return false;
  }

  /// isFunctionDeclarator - This method returns true if the declarator
  /// is a function declarator (looking through parentheses).
  /// If true is returned, then the reference type parameter idx is
  /// assigned with the index of the declaration chunk.
  bool isFunctionDeclarator(unsigned& idx) const {
    for (unsigned i = 0, i_end = DeclTypeInfo.size(); i < i_end; ++i) {
      switch (DeclTypeInfo[i].Kind) {
      case DeclaratorChunk::Function:
        idx = i;
        return true;
      case DeclaratorChunk::Paren:
        continue;
      case DeclaratorChunk::Pointer:
      case DeclaratorChunk::Reference:
      case DeclaratorChunk::Array:
      case DeclaratorChunk::BlockPointer:
      case DeclaratorChunk::MemberPointer:
        return false;
      }
      llvm_unreachable("Invalid type chunk");
    }
    return false;
  }

  /// isFunctionDeclarator - Once this declarator is fully parsed and formed,
  /// this method returns true if the identifier is a function declarator
  /// (looking through parentheses).
  bool isFunctionDeclarator() const {
    unsigned index;
    return isFunctionDeclarator(index);
  }

  /// getFunctionTypeInfo - Retrieves the function type info object
  /// (looking through parentheses).
  DeclaratorChunk::FunctionTypeInfo &getFunctionTypeInfo() {
    assert(isFunctionDeclarator() && "Not a function declarator!");
    unsigned index = 0;
    isFunctionDeclarator(index);
    return DeclTypeInfo[index].Fun;
  }

  /// getFunctionTypeInfo - Retrieves the function type info object
  /// (looking through parentheses).
  const DeclaratorChunk::FunctionTypeInfo &getFunctionTypeInfo() const {
    return const_cast<Declarator*>(this)->getFunctionTypeInfo();
  }

  /// \brief Determine whether the declaration that will be produced from 
  /// this declaration will be a function.
  /// 
  /// A declaration can declare a function even if the declarator itself
  /// isn't a function declarator, if the type specifier refers to a function
  /// type. This routine checks for both cases.
  bool isDeclarationOfFunction() const;
  
  /// takeAttributes - Takes attributes from the given parsed-attributes
  /// set and add them to this declarator.
  ///
  /// These examples both add 3 attributes to "var":
  ///  short int var __attribute__((aligned(16),common,deprecated));
  ///  short int x, __attribute__((aligned(16)) var
  ///                                 __attribute__((common,deprecated));
  ///
  /// Also extends the range of the declarator.
  void takeAttributes(ParsedAttributes &attrs, SourceLocation lastLoc) {
    Attrs.takeAllFrom(attrs);

    if (!lastLoc.isInvalid())
      SetRangeEnd(lastLoc);
  }

  const AttributeList *getAttributes() const { return Attrs.getList(); }
  AttributeList *getAttributes() { return Attrs.getList(); }

  AttributeList *&getAttrListRef() { return Attrs.getListRef(); }

  /// hasAttributes - do we contain any attributes?
  bool hasAttributes() const {
    if (getAttributes() || getDeclSpec().hasAttributes()) return true;
    for (unsigned i = 0, e = getNumTypeObjects(); i != e; ++i)
      if (getTypeObject(i).getAttrs())
        return true;
    return false;
  }

  void setAsmLabel(Expr *E) { AsmLabel = E; }
  Expr *getAsmLabel() const { return AsmLabel; }

  void setExtension(bool Val = true) { Extension = Val; }
  bool getExtension() const { return Extension; }

  void setInvalidType(bool Val = true) { InvalidType = Val; }
  bool isInvalidType() const {
    return InvalidType || DS.getTypeSpecType() == DeclSpec::TST_error;
  }

  void setGroupingParens(bool flag) { GroupingParens = flag; }
  bool hasGroupingParens() const { return GroupingParens; }

  bool isFirstDeclarator() const { return !CommaLoc.isValid(); }
  SourceLocation getCommaLoc() const { return CommaLoc; }
  void setCommaLoc(SourceLocation CL) { CommaLoc = CL; }

  bool hasEllipsis() const { return EllipsisLoc.isValid(); }
  SourceLocation getEllipsisLoc() const { return EllipsisLoc; }
  void setEllipsisLoc(SourceLocation EL) { EllipsisLoc = EL; }

  void setFunctionDefinitionKind(FunctionDefinitionKind Val) { 
    FunctionDefinition = Val; 
  }
  
  bool isFunctionDefinition() const {
    return getFunctionDefinitionKind() != FDK_Declaration;
  }
  
  FunctionDefinitionKind getFunctionDefinitionKind() const { 
    return (FunctionDefinitionKind)FunctionDefinition; 
  }

  void setRedeclaration(bool Val) { Redeclaration = Val; }
  bool isRedeclaration() const { return Redeclaration; }
};

/// \brief This little struct is used to capture information about
/// structure field declarators, which is basically just a bitfield size.
struct FieldDeclarator {
  Declarator D;
  Expr *BitfieldSize;
  explicit FieldDeclarator(const DeclSpec &DS)
    : D(DS, Declarator::MemberContext), BitfieldSize(0) { }
};

/// \brief Represents a C++11 virt-specifier-seq.
class VirtSpecifiers {
public:
  enum Specifier {
    VS_None = 0,
    VS_Override = 1,
    VS_Final = 2
  };

  VirtSpecifiers() : Specifiers(0) { }

  bool SetSpecifier(Specifier VS, SourceLocation Loc,
                    const char *&PrevSpec);

  bool isOverrideSpecified() const { return Specifiers & VS_Override; }
  SourceLocation getOverrideLoc() const { return VS_overrideLoc; }

  bool isFinalSpecified() const { return Specifiers & VS_Final; }
  SourceLocation getFinalLoc() const { return VS_finalLoc; }

  void clear() { Specifiers = 0; }

  static const char *getSpecifierName(Specifier VS);

  SourceLocation getLastLocation() const { return LastLocation; }
  
private:
  unsigned Specifiers;

  SourceLocation VS_overrideLoc, VS_finalLoc;
  SourceLocation LastLocation;
};

/// \brief An individual capture in a lambda introducer.
struct LambdaCapture {
  LambdaCaptureKind Kind;
  SourceLocation Loc;
  IdentifierInfo* Id;
  SourceLocation EllipsisLoc;
  
  LambdaCapture(LambdaCaptureKind Kind, SourceLocation Loc,
                IdentifierInfo* Id = 0,
                SourceLocation EllipsisLoc = SourceLocation())
    : Kind(Kind), Loc(Loc), Id(Id), EllipsisLoc(EllipsisLoc)
  {}
};

/// \brief Represents a complete lambda introducer.
struct LambdaIntroducer {
  SourceRange Range;
  SourceLocation DefaultLoc;
  LambdaCaptureDefault Default;
  llvm::SmallVector<LambdaCapture, 4> Captures;

  LambdaIntroducer()
    : Default(LCD_None) {}

  /// \brief Append a capture in a lambda introducer.
  void addCapture(LambdaCaptureKind Kind,
                  SourceLocation Loc,
                  IdentifierInfo* Id = 0, 
                  SourceLocation EllipsisLoc = SourceLocation()) {
    Captures.push_back(LambdaCapture(Kind, Loc, Id, EllipsisLoc));
  }

};

} // end namespace clang

#endif
