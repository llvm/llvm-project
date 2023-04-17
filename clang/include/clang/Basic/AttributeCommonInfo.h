//======- AttributeCommonInfo.h - Base info about Attributes-----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AttributeCommonInfo type, which is the base for a
// ParsedAttr and is used by Attr as a way to share info between the two.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_ATTRIBUTECOMMONINFO_H
#define LLVM_CLANG_BASIC_ATTRIBUTECOMMONINFO_H
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TokenKinds.h"

namespace clang {
class IdentifierInfo;
class ASTRecordWriter;

class AttributeCommonInfo {
public:
  /// The style used to specify an attribute.
  enum Syntax {
    /// __attribute__((...))
    AS_GNU,

    /// [[...]]
    AS_CXX11,

    /// [[...]]
    AS_C2x,

    /// __declspec(...)
    AS_Declspec,

    /// [uuid("...")] class Foo
    AS_Microsoft,

    /// __ptr16, alignas(...), etc.
    AS_Keyword,

    /// #pragma ...
    AS_Pragma,

    // Note TableGen depends on the order above.  Do not add or change the order
    // without adding related code to TableGen/ClangAttrEmitter.cpp.
    /// Context-sensitive version of a keyword attribute.
    AS_ContextSensitiveKeyword,

    /// <vardecl> : <semantic>
    AS_HLSLSemantic,

    /// The attibute has no source code manifestation and is only created
    /// implicitly.
    AS_Implicit
  };
  enum Kind {
#define PARSED_ATTR(NAME) AT_##NAME,
#include "clang/Sema/AttrParsedAttrList.inc"
#undef PARSED_ATTR
    NoSemaHandlerAttribute,
    IgnoredAttribute,
    UnknownAttribute,
  };

private:
  const IdentifierInfo *AttrName = nullptr;
  const IdentifierInfo *ScopeName = nullptr;
  SourceRange AttrRange;
  const SourceLocation ScopeLoc;
  // Corresponds to the Kind enum.
  unsigned AttrKind : 16;
  /// Corresponds to the Syntax enum.
  unsigned SyntaxUsed : 4;
  unsigned SpellingIndex : 4;
  unsigned IsAlignas : 1;

protected:
  static constexpr unsigned SpellingNotCalculated = 0xf;

public:
  /// Combines information about the source-code form of an attribute,
  /// including its syntax and spelling.
  class Form {
  public:
    constexpr Form(Syntax SyntaxUsed, unsigned SpellingIndex, bool IsAlignas)
        : SyntaxUsed(SyntaxUsed), SpellingIndex(SpellingIndex),
          IsAlignas(IsAlignas) {}
    constexpr Form(tok::TokenKind Tok)
        : SyntaxUsed(AS_Keyword), SpellingIndex(SpellingNotCalculated),
          IsAlignas(Tok == tok::kw_alignas) {}

    Syntax getSyntax() const { return Syntax(SyntaxUsed); }
    unsigned getSpellingIndex() const { return SpellingIndex; }
    bool isAlignas() const { return IsAlignas; }

    static Form GNU() { return AS_GNU; }
    static Form CXX11() { return AS_CXX11; }
    static Form C2x() { return AS_C2x; }
    static Form Declspec() { return AS_Declspec; }
    static Form Microsoft() { return AS_Microsoft; }
    static Form Keyword(bool IsAlignas) {
      return Form(AS_Keyword, SpellingNotCalculated, IsAlignas);
    }
    static Form Pragma() { return AS_Pragma; }
    static Form ContextSensitiveKeyword() { return AS_ContextSensitiveKeyword; }
    static Form HLSLSemantic() { return AS_HLSLSemantic; }
    static Form Implicit() { return AS_Implicit; }

  private:
    constexpr Form(Syntax SyntaxUsed)
        : SyntaxUsed(SyntaxUsed), SpellingIndex(SpellingNotCalculated),
          IsAlignas(0) {}

    unsigned SyntaxUsed : 4;
    unsigned SpellingIndex : 4;
    unsigned IsAlignas : 1;
  };

  AttributeCommonInfo(const IdentifierInfo *AttrName,
                      const IdentifierInfo *ScopeName, SourceRange AttrRange,
                      SourceLocation ScopeLoc, Form FormUsed)
      : AttrName(AttrName), ScopeName(ScopeName), AttrRange(AttrRange),
        ScopeLoc(ScopeLoc),
        AttrKind(getParsedKind(AttrName, ScopeName, FormUsed.getSyntax())),
        SyntaxUsed(FormUsed.getSyntax()),
        SpellingIndex(FormUsed.getSpellingIndex()),
        IsAlignas(FormUsed.isAlignas()) {}

  AttributeCommonInfo(const IdentifierInfo *AttrName,
                      const IdentifierInfo *ScopeName, SourceRange AttrRange,
                      SourceLocation ScopeLoc, Kind AttrKind, Form FormUsed)
      : AttrName(AttrName), ScopeName(ScopeName), AttrRange(AttrRange),
        ScopeLoc(ScopeLoc), AttrKind(AttrKind),
        SyntaxUsed(FormUsed.getSyntax()),
        SpellingIndex(FormUsed.getSpellingIndex()),
        IsAlignas(FormUsed.isAlignas()) {}

  AttributeCommonInfo(const IdentifierInfo *AttrName, SourceRange AttrRange,
                      Form FormUsed)
      : AttrName(AttrName), ScopeName(nullptr), AttrRange(AttrRange),
        ScopeLoc(),
        AttrKind(getParsedKind(AttrName, ScopeName, FormUsed.getSyntax())),
        SyntaxUsed(FormUsed.getSyntax()),
        SpellingIndex(FormUsed.getSpellingIndex()),
        IsAlignas(FormUsed.isAlignas()) {}

  AttributeCommonInfo(SourceRange AttrRange, Kind K, Form FormUsed)
      : AttrName(nullptr), ScopeName(nullptr), AttrRange(AttrRange), ScopeLoc(),
        AttrKind(K), SyntaxUsed(FormUsed.getSyntax()),
        SpellingIndex(FormUsed.getSpellingIndex()),
        IsAlignas(FormUsed.isAlignas()) {}

  AttributeCommonInfo(AttributeCommonInfo &&) = default;
  AttributeCommonInfo(const AttributeCommonInfo &) = default;

  Kind getParsedKind() const { return Kind(AttrKind); }
  Syntax getSyntax() const { return Syntax(SyntaxUsed); }
  Form getForm() const { return Form(getSyntax(), SpellingIndex, IsAlignas); }
  const IdentifierInfo *getAttrName() const { return AttrName; }
  SourceLocation getLoc() const { return AttrRange.getBegin(); }
  SourceRange getRange() const { return AttrRange; }
  void setRange(SourceRange R) { AttrRange = R; }

  bool hasScope() const { return ScopeName; }
  const IdentifierInfo *getScopeName() const { return ScopeName; }
  SourceLocation getScopeLoc() const { return ScopeLoc; }

  /// Gets the normalized full name, which consists of both scope and name and
  /// with surrounding underscores removed as appropriate (e.g.
  /// __gnu__::__attr__ will be normalized to gnu::attr).
  std::string getNormalizedFullName() const;

  bool isDeclspecAttribute() const { return SyntaxUsed == AS_Declspec; }
  bool isMicrosoftAttribute() const { return SyntaxUsed == AS_Microsoft; }

  bool isGNUScope() const;
  bool isClangScope() const;

  bool isCXX11Attribute() const { return SyntaxUsed == AS_CXX11 || IsAlignas; }

  bool isC2xAttribute() const { return SyntaxUsed == AS_C2x; }

  /// The attribute is spelled [[]] in either C or C++ mode, including standard
  /// attributes spelled with a keyword, like alignas.
  bool isStandardAttributeSyntax() const {
    return isCXX11Attribute() || isC2xAttribute();
  }

  bool isGNUAttribute() const { return SyntaxUsed == AS_GNU; }

  bool isKeywordAttribute() const {
    return SyntaxUsed == AS_Keyword || SyntaxUsed == AS_ContextSensitiveKeyword;
  }

  bool isContextSensitiveKeywordAttribute() const {
    return SyntaxUsed == AS_ContextSensitiveKeyword;
  }

  unsigned getAttributeSpellingListIndex() const {
    assert((isAttributeSpellingListCalculated() || AttrName) &&
           "Spelling cannot be found");
    return isAttributeSpellingListCalculated()
               ? SpellingIndex
               : calculateAttributeSpellingListIndex();
  }
  void setAttributeSpellingListIndex(unsigned V) { SpellingIndex = V; }

  static Kind getParsedKind(const IdentifierInfo *Name,
                            const IdentifierInfo *Scope, Syntax SyntaxUsed);

private:
  /// Get an index into the attribute spelling list
  /// defined in Attr.td. This index is used by an attribute
  /// to pretty print itself.
  unsigned calculateAttributeSpellingListIndex() const;

  friend class clang::ASTRecordWriter;
  // Used exclusively by ASTDeclWriter to get the raw spelling list state.
  unsigned getAttributeSpellingListIndexRaw() const { return SpellingIndex; }

protected:
  bool isAttributeSpellingListCalculated() const {
    return SpellingIndex != SpellingNotCalculated;
  }
};
} // namespace clang

#endif // LLVM_CLANG_BASIC_ATTRIBUTECOMMONINFO_H
