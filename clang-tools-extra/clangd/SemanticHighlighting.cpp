//===--- SemanticHighlighting.cpp - ------------------------- ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SemanticHighlighting.h"
#include "Logger.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"

namespace clang {
namespace clangd {
namespace {

// Collects all semantic tokens in an ASTContext.
class HighlightingTokenCollector
    : public RecursiveASTVisitor<HighlightingTokenCollector> {
  std::vector<HighlightingToken> Tokens;
  ASTContext &Ctx;
  const SourceManager &SM;

public:
  HighlightingTokenCollector(ParsedAST &AST)
      : Ctx(AST.getASTContext()), SM(AST.getSourceManager()) {}

  std::vector<HighlightingToken> collectTokens() {
    Tokens.clear();
    TraverseAST(Ctx);
    // Initializer lists can give duplicates of tokens, therefore all tokens
    // must be deduplicated.
    llvm::sort(Tokens,
               [](const HighlightingToken &L, const HighlightingToken &R) {
                 return std::tie(L.R, L.Kind) < std::tie(R.R, R.Kind);
               });
    auto Last = std::unique(Tokens.begin(), Tokens.end());
    Tokens.erase(Last, Tokens.end());
    return Tokens;
  }

  bool VisitNamespaceAliasDecl(NamespaceAliasDecl *NAD) {
    // The target namespace of an alias can not be found in any other way.
    addToken(NAD->getTargetNameLoc(), HighlightingKind::Namespace);
    return true;
  }

  bool VisitMemberExpr(MemberExpr *ME) {
    const auto *MD = ME->getMemberDecl();
    if (isa<CXXDestructorDecl>(MD))
      // When calling the destructor manually like: AAA::~A(); The ~ is a
      // MemberExpr. Other methods should still be highlighted though.
      return true;
    addToken(ME->getMemberLoc(), MD);
    return true;
  }

  bool VisitNamedDecl(NamedDecl *ND) {
    // UsingDirectiveDecl's namespaces do not show up anywhere else in the
    // Visit/Traverse mehods. But they should also be highlighted as a
    // namespace.
    if (const auto *UD = dyn_cast<UsingDirectiveDecl>(ND)) {
      addToken(UD->getIdentLocation(), HighlightingKind::Namespace);
      return true;
    }

    // Constructors' TypeLoc has a TypePtr that is a FunctionProtoType. It has
    // no tag decl and therefore constructors must be gotten as NamedDecls
    // instead.
    if (ND->getDeclName().getNameKind() ==
        DeclarationName::CXXConstructorName) {
      addToken(ND->getLocation(), ND);
      return true;
    }

    if (ND->getDeclName().getNameKind() != DeclarationName::Identifier)
      return true;

    addToken(ND->getLocation(), ND);
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *Ref) {
    if (Ref->getNameInfo().getName().getNameKind() !=
        DeclarationName::Identifier)
      // Only want to highlight identifiers.
      return true;

    addToken(Ref->getLocation(), Ref->getDecl());
    return true;
  }

  bool VisitTypedefNameDecl(TypedefNameDecl *TD) {
    if(const auto *TSI = TD->getTypeSourceInfo())
      addTypeLoc(TD->getLocation(), TSI->getTypeLoc());
    return true;
  }

  bool VisitTemplateTypeParmTypeLoc(TemplateTypeParmTypeLoc &TL) {
    // TemplateTypeParmTypeLoc does not have a TagDecl in its type ptr.
    addToken(TL.getBeginLoc(), TL.getDecl());
    return true;
  }

  bool VisitTemplateSpecializationTypeLoc(TemplateSpecializationTypeLoc &TL) {
    if (const TemplateDecl *TD =
            TL.getTypePtr()->getTemplateName().getAsTemplateDecl())
      addToken(TL.getBeginLoc(), TD);
    return true;
  }

  bool VisitTypeLoc(TypeLoc &TL) {
    // This check is for not getting two entries when there are anonymous
    // structs. It also makes us not highlight certain namespace qualifiers
    // twice. For elaborated types the actual type is highlighted as an inner
    // TypeLoc.
    if (TL.getTypeLocClass() == TypeLoc::TypeLocClass::Elaborated)
      return true;

    addTypeLoc(TL.getBeginLoc(), TL);
    return true;
  }

  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNSLoc) {
    if (NestedNameSpecifier *NNS = NNSLoc.getNestedNameSpecifier())
      if (NNS->getKind() == NestedNameSpecifier::Namespace ||
          NNS->getKind() == NestedNameSpecifier::NamespaceAlias)
        addToken(NNSLoc.getLocalBeginLoc(), HighlightingKind::Namespace);

    return RecursiveASTVisitor<
        HighlightingTokenCollector>::TraverseNestedNameSpecifierLoc(NNSLoc);
  }

private:
  void addTypeLoc(SourceLocation Loc, const TypeLoc &TL) {
    if (const Type *TP = TL.getTypePtr())
      if (const TagDecl *TD = TP->getAsTagDecl())
        addToken(Loc, TD);
  }

  void addToken(SourceLocation Loc, const NamedDecl *D) {
    if (D->getDeclName().isIdentifier() && D->getName().empty())
      // Don't add symbols that don't have any length.
      return;
    // We highlight class decls, constructor decls and destructor decls as
    // `Class` type. The destructor decls are handled in `VisitTypeLoc` (we will
    // visit a TypeLoc where the underlying Type is a CXXRecordDecl).
    if (isa<ClassTemplateDecl>(D)) {
      addToken(Loc, HighlightingKind::Class);
      return;
    }
    if (isa<RecordDecl>(D)) {
      addToken(Loc, HighlightingKind::Class);
      return;
    }
    if (isa<CXXConstructorDecl>(D)) {
      addToken(Loc, HighlightingKind::Class);
      return;
    }
    if (isa<CXXMethodDecl>(D)) {
      addToken(Loc, HighlightingKind::Method);
      return;
    }
    if (isa<FieldDecl>(D)) {
      addToken(Loc, HighlightingKind::Field);
      return;
    }
    if (isa<EnumDecl>(D)) {
      addToken(Loc, HighlightingKind::Enum);
      return;
    }
    if (isa<EnumConstantDecl>(D)) {
      addToken(Loc, HighlightingKind::EnumConstant);
      return;
    }
    if (isa<VarDecl>(D)) {
      addToken(Loc, HighlightingKind::Variable);
      return;
    }
    if (isa<FunctionDecl>(D)) {
      addToken(Loc, HighlightingKind::Function);
      return;
    }
    if (isa<NamespaceDecl>(D)) {
      addToken(Loc, HighlightingKind::Namespace);
      return;
    }
    if (isa<NamespaceAliasDecl>(D)) {
      addToken(Loc, HighlightingKind::Namespace);
      return;
    }
    if (isa<TemplateTemplateParmDecl>(D)) {
      addToken(Loc, HighlightingKind::TemplateParameter);
      return;
    }
    if (isa<TemplateTypeParmDecl>(D)) {
      addToken(Loc, HighlightingKind::TemplateParameter);
      return;
    }
  }

  void addToken(SourceLocation Loc, HighlightingKind Kind) {
    if (Loc.isMacroID())
      // FIXME: skip tokens inside macros for now.
      return;

    auto R = getTokenRange(SM, Ctx.getLangOpts(), Loc);
    if (!R) {
      // R should always have a value, if it doesn't something is very wrong.
      elog("Tried to add semantic token with an invalid range");
      return;
    }

    Tokens.push_back({Kind, R.getValue()});
  }
};

// Encode binary data into base64.
// This was copied from compiler-rt/lib/fuzzer/FuzzerUtil.cpp.
// FIXME: Factor this out into llvm/Support?
std::string encodeBase64(const llvm::SmallVectorImpl<char> &Bytes) {
  static const char Table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                              "abcdefghijklmnopqrstuvwxyz"
                              "0123456789+/";
  std::string Res;
  size_t I;
  for (I = 0; I + 2 < Bytes.size(); I += 3) {
    uint32_t X = (Bytes[I] << 16) + (Bytes[I + 1] << 8) + Bytes[I + 2];
    Res += Table[(X >> 18) & 63];
    Res += Table[(X >> 12) & 63];
    Res += Table[(X >> 6) & 63];
    Res += Table[X & 63];
  }
  if (I + 1 == Bytes.size()) {
    uint32_t X = (Bytes[I] << 16);
    Res += Table[(X >> 18) & 63];
    Res += Table[(X >> 12) & 63];
    Res += "==";
  } else if (I + 2 == Bytes.size()) {
    uint32_t X = (Bytes[I] << 16) + (Bytes[I + 1] << 8);
    Res += Table[(X >> 18) & 63];
    Res += Table[(X >> 12) & 63];
    Res += Table[(X >> 6) & 63];
    Res += "=";
  }
  return Res;
}

void write32be(uint32_t I, llvm::raw_ostream &OS) {
  std::array<char, 4> Buf;
  llvm::support::endian::write32be(Buf.data(), I);
  OS.write(Buf.data(), Buf.size());
}

void write16be(uint16_t I, llvm::raw_ostream &OS) {
  std::array<char, 2> Buf;
  llvm::support::endian::write16be(Buf.data(), I);
  OS.write(Buf.data(), Buf.size());
}
} // namespace

bool operator==(const HighlightingToken &Lhs, const HighlightingToken &Rhs) {
  return Lhs.Kind == Rhs.Kind && Lhs.R == Rhs.R;
}

std::vector<HighlightingToken> getSemanticHighlightings(ParsedAST &AST) {
  return HighlightingTokenCollector(AST).collectTokens();
}

std::vector<SemanticHighlightingInformation>
toSemanticHighlightingInformation(llvm::ArrayRef<HighlightingToken> Tokens) {
  if (Tokens.size() == 0)
    return {};

  // FIXME: Tokens might be multiple lines long (block comments) in this case
  // this needs to add multiple lines for those tokens.
  std::map<int, std::vector<HighlightingToken>> TokenLines;
  for (const HighlightingToken &Token : Tokens)
    TokenLines[Token.R.start.line].push_back(Token);

  std::vector<SemanticHighlightingInformation> Lines;
  Lines.reserve(TokenLines.size());
  for (const auto &Line : TokenLines) {
    llvm::SmallVector<char, 128> LineByteTokens;
    llvm::raw_svector_ostream OS(LineByteTokens);
    for (const auto &Token : Line.second) {
      // Writes the token to LineByteTokens in the byte format specified by the
      // LSP proposal. Described below.
      // |<---- 4 bytes ---->|<-- 2 bytes -->|<--- 2 bytes -->|
      // |    character      |  length       |    index       |

      write32be(Token.R.start.character, OS);
      write16be(Token.R.end.character - Token.R.start.character, OS);
      write16be(static_cast<int>(Token.Kind), OS);
    }

    Lines.push_back({Line.first, encodeBase64(LineByteTokens)});
  }

  return Lines;
}

llvm::StringRef toTextMateScope(HighlightingKind Kind) {
  // FIXME: Add scopes for C and Objective C.
  switch (Kind) {
  case HighlightingKind::Function:
    return "entity.name.function.cpp";
  case HighlightingKind::Method:
    return "entity.name.function.method.cpp";
  case HighlightingKind::Variable:
    return "variable.other.cpp";
  case HighlightingKind::Field:
    return "variable.other.field.cpp";
  case HighlightingKind::Class:
    return "entity.name.type.class.cpp";
  case HighlightingKind::Enum:
    return "entity.name.type.enum.cpp";
  case HighlightingKind::EnumConstant:
    return "variable.other.enummember.cpp";
  case HighlightingKind::Namespace:
    return "entity.name.namespace.cpp";
  case HighlightingKind::TemplateParameter:
    return "entity.name.type.template.cpp";
  case HighlightingKind::NumKinds:
    llvm_unreachable("must not pass NumKinds to the function");
  }
  llvm_unreachable("unhandled HighlightingKind");
}

} // namespace clangd
} // namespace clang
