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
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include <algorithm>

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
    llvm::sort(Tokens);
    auto Last = std::unique(Tokens.begin(), Tokens.end());
    Tokens.erase(Last, Tokens.end());
    // Macros can give tokens that have the same source range but conflicting
    // kinds. In this case all tokens sharing this source range should be
    // removed.
    std::vector<HighlightingToken> NonConflicting;
    NonConflicting.reserve(Tokens.size());
    for (ArrayRef<HighlightingToken> TokRef = Tokens; !TokRef.empty();) {
      ArrayRef<HighlightingToken> Conflicting =
          TokRef.take_while([&](const HighlightingToken &T) {
            // TokRef is guaranteed at least one element here because otherwise
            // this predicate would never fire.
            return T.R == TokRef.front().R;
          });
      // If there is exactly one token with this range it's non conflicting and
      // should be in the highlightings.
      if (Conflicting.size() == 1)
        NonConflicting.push_back(TokRef.front());
      // TokRef[Conflicting.size()] is the next token with a different range (or
      // the end of the Tokens).
      TokRef = TokRef.drop_front(Conflicting.size());
    }
    return NonConflicting;
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
    if (isa<CXXConversionDecl>(MD))
      // The MemberLoc is invalid for C++ conversion operators. We do not
      // attempt to add tokens with invalid locations.
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
    if (const auto *TSI = TD->getTypeSourceInfo())
      addType(TD->getLocation(), TSI->getTypeLoc().getTypePtr());
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
    if (TL.getTypeLocClass() != TypeLoc::TypeLocClass::Elaborated)
      addType(TL.getBeginLoc(), TL.getTypePtr());
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

  bool TraverseConstructorInitializer(CXXCtorInitializer *CI) {
    if (const FieldDecl *FD = CI->getMember())
      addToken(CI->getSourceLocation(), FD);
    return RecursiveASTVisitor<
        HighlightingTokenCollector>::TraverseConstructorInitializer(CI);
  }

  bool VisitDeclaratorDecl(DeclaratorDecl *D) {
    if ((!D->getTypeSourceInfo()))
      return true;

    if (auto *AT = D->getType()->getContainedAutoType()) {
      const auto Deduced = AT->getDeducedType();
      if (!Deduced.isNull())
        addType(D->getTypeSpecStartLoc(), Deduced.getTypePtr());
    }
    return true;
  }

private:
  void addType(SourceLocation Loc, const Type *TP) {
    if (!TP)
      return;
    if (TP->isBuiltinType())
      // Builtins must be special cased as they do not have a TagDecl.
      addToken(Loc, HighlightingKind::Primitive);
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
    if (isa<ParmVarDecl>(D)) {
      addToken(Loc, HighlightingKind::Parameter);
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
    if (isa<NonTypeTemplateParmDecl>(D)) {
      addToken(Loc, HighlightingKind::TemplateParameter);
      return;
    }
  }

  void addToken(SourceLocation Loc, HighlightingKind Kind) {
    if(Loc.isMacroID()) {
      // Only intereseted in highlighting arguments in macros (DEF_X(arg)).
      if (!SM.isMacroArgExpansion(Loc))
        return;
      Loc = SM.getSpellingLoc(Loc);
    }

    // Non top level decls that are included from a header are not filtered by
    // topLevelDecls. (example: method declarations being included from another
    // file for a class from another file)
    // There are also cases with macros where the spelling loc will not be in the
    // main file and the highlighting would be incorrect.
    if (!isInsideMainFile(Loc, SM))
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

// Get the highlightings on \c Line where the first entry of line is at \c
// StartLineIt. If it is not at \c StartLineIt an empty vector is returned.
ArrayRef<HighlightingToken>
takeLine(ArrayRef<HighlightingToken> AllTokens,
         ArrayRef<HighlightingToken>::iterator StartLineIt, int Line) {
  return ArrayRef<HighlightingToken>(StartLineIt, AllTokens.end())
      .take_while([Line](const HighlightingToken &Token) {
        return Token.R.start.line == Line;
      });
}
} // namespace

std::vector<LineHighlightings>
diffHighlightings(ArrayRef<HighlightingToken> New,
                  ArrayRef<HighlightingToken> Old, int NewMaxLine) {
  assert(std::is_sorted(New.begin(), New.end()) && "New must be a sorted vector");
  assert(std::is_sorted(Old.begin(), Old.end()) && "Old must be a sorted vector");

  // FIXME: There's an edge case when tokens span multiple lines. If the first
  // token on the line started on a line above the current one and the rest of
  // the line is the equal to the previous one than we will remove all
  // highlights but the ones for the token spanning multiple lines. This means
  // that when we get into the LSP layer the only highlights that will be
  // visible are the ones for the token spanning multiple lines.
  // Example:
  // EndOfMultilineToken  Token Token Token
  // If "Token Token Token" don't differ from previously the line is
  // incorrectly removed. Suggestion to fix is to separate any multiline tokens
  // into one token for every line it covers. This requires reading from the
  // file buffer to figure out the length of each line though.
  std::vector<LineHighlightings> DiffedLines;
  // ArrayRefs to the current line in the highlightings.
  ArrayRef<HighlightingToken> NewLine(New.begin(),
                                      /*length*/ static_cast<size_t>(0));
  ArrayRef<HighlightingToken> OldLine(Old.begin(),
                                      /*length*/ static_cast<size_t>(0));
  auto NewEnd = New.end();
  auto OldEnd = Old.end();
  auto NextLineNumber = [&]() {
    int NextNew = NewLine.end() != NewEnd ? NewLine.end()->R.start.line
                                             : std::numeric_limits<int>::max();
    int NextOld = OldLine.end() != OldEnd ? OldLine.end()->R.start.line
                                             : std::numeric_limits<int>::max();
    return std::min(NextNew, NextOld);
  };

  // If the New file has fewer lines than the Old file we don't want to send
  // highlightings beyond the end of the file.
  for (int LineNumber = 0; LineNumber < NewMaxLine;
       LineNumber = NextLineNumber()) {
    NewLine = takeLine(New, NewLine.end(), LineNumber);
    OldLine = takeLine(Old, OldLine.end(), LineNumber);
    if (NewLine != OldLine)
      DiffedLines.push_back({LineNumber, NewLine});
  }

  return DiffedLines;
}

bool operator==(const HighlightingToken &L, const HighlightingToken &R) {
  return std::tie(L.R, L.Kind) == std::tie(R.R, R.Kind);
}
bool operator<(const HighlightingToken &L, const HighlightingToken &R) {
  return std::tie(L.R, L.Kind) < std::tie(R.R, R.Kind);
}
bool operator==(const LineHighlightings &L, const LineHighlightings &R) {
  return std::tie(L.Line, L.Tokens) == std::tie(R.Line, R.Tokens);
}

std::vector<HighlightingToken> getSemanticHighlightings(ParsedAST &AST) {
  return HighlightingTokenCollector(AST).collectTokens();
}

std::vector<SemanticHighlightingInformation>
toSemanticHighlightingInformation(llvm::ArrayRef<LineHighlightings> Tokens) {
  if (Tokens.size() == 0)
    return {};

  // FIXME: Tokens might be multiple lines long (block comments) in this case
  // this needs to add multiple lines for those tokens.
  std::vector<SemanticHighlightingInformation> Lines;
  Lines.reserve(Tokens.size());
  for (const auto &Line : Tokens) {
    llvm::SmallVector<char, 128> LineByteTokens;
    llvm::raw_svector_ostream OS(LineByteTokens);
    for (const auto &Token : Line.Tokens) {
      // Writes the token to LineByteTokens in the byte format specified by the
      // LSP proposal. Described below.
      // |<---- 4 bytes ---->|<-- 2 bytes -->|<--- 2 bytes -->|
      // |    character      |  length       |    index       |

      write32be(Token.R.start.character, OS);
      write16be(Token.R.end.character - Token.R.start.character, OS);
      write16be(static_cast<int>(Token.Kind), OS);
    }

    Lines.push_back({Line.Line, encodeBase64(LineByteTokens)});
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
  case HighlightingKind::Parameter:
    return "variable.parameter.cpp";
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
  case HighlightingKind::Primitive:
    return "storage.type.primitive.cpp";
  case HighlightingKind::NumKinds:
    llvm_unreachable("must not pass NumKinds to the function");
  }
  llvm_unreachable("unhandled HighlightingKind");
}

} // namespace clangd
} // namespace clang
