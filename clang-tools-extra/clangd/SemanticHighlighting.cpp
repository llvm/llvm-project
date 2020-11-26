//===--- SemanticHighlighting.cpp - ------------------------- ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SemanticHighlighting.h"
#include "FindTarget.h"
#include "ParsedAST.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "support/Logger.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Casting.h"
#include <algorithm>

namespace clang {
namespace clangd {
namespace {

/// Some names are not written in the source code and cannot be highlighted,
/// e.g. anonymous classes. This function detects those cases.
bool canHighlightName(DeclarationName Name) {
  if (Name.getNameKind() == DeclarationName::CXXConstructorName ||
      Name.getNameKind() == DeclarationName::CXXUsingDirective)
    return true;
  auto *II = Name.getAsIdentifierInfo();
  return II && !II->getName().empty();
}

llvm::Optional<HighlightingKind> kindForType(const Type *TP);
llvm::Optional<HighlightingKind> kindForDecl(const NamedDecl *D) {
  if (auto *USD = dyn_cast<UsingShadowDecl>(D)) {
    if (auto *Target = USD->getTargetDecl())
      D = Target;
  }
  if (auto *TD = dyn_cast<TemplateDecl>(D)) {
    if (auto *Templated = TD->getTemplatedDecl())
      D = Templated;
  }
  if (auto *TD = dyn_cast<TypedefNameDecl>(D)) {
    // We try to highlight typedefs as their underlying type.
    if (auto K = kindForType(TD->getUnderlyingType().getTypePtrOrNull()))
      return K;
    // And fallback to a generic kind if this fails.
    return HighlightingKind::Typedef;
  }
  // We highlight class decls, constructor decls and destructor decls as
  // `Class` type. The destructor decls are handled in `VisitTagTypeLoc` (we
  // will visit a TypeLoc where the underlying Type is a CXXRecordDecl).
  if (auto *RD = llvm::dyn_cast<RecordDecl>(D)) {
    // We don't want to highlight lambdas like classes.
    if (RD->isLambda())
      return llvm::None;
    return HighlightingKind::Class;
  }
  if (isa<ClassTemplateDecl>(D) || isa<RecordDecl>(D) ||
      isa<CXXConstructorDecl>(D))
    return HighlightingKind::Class;
  if (auto *MD = dyn_cast<CXXMethodDecl>(D))
    return MD->isStatic() ? HighlightingKind::StaticMethod
                          : HighlightingKind::Method;
  if (isa<FieldDecl>(D))
    return HighlightingKind::Field;
  if (isa<EnumDecl>(D))
    return HighlightingKind::Enum;
  if (isa<EnumConstantDecl>(D))
    return HighlightingKind::EnumConstant;
  if (isa<ParmVarDecl>(D))
    return HighlightingKind::Parameter;
  if (auto *VD = dyn_cast<VarDecl>(D))
    return VD->isStaticDataMember()
               ? HighlightingKind::StaticField
               : VD->isLocalVarDecl() ? HighlightingKind::LocalVariable
                                      : HighlightingKind::Variable;
  if (const auto *BD = dyn_cast<BindingDecl>(D))
    return BD->getDeclContext()->isFunctionOrMethod()
               ? HighlightingKind::LocalVariable
               : HighlightingKind::Variable;
  if (isa<FunctionDecl>(D))
    return HighlightingKind::Function;
  if (isa<NamespaceDecl>(D) || isa<NamespaceAliasDecl>(D) ||
      isa<UsingDirectiveDecl>(D))
    return HighlightingKind::Namespace;
  if (isa<TemplateTemplateParmDecl>(D) || isa<TemplateTypeParmDecl>(D) ||
      isa<NonTypeTemplateParmDecl>(D))
    return HighlightingKind::TemplateParameter;
  if (isa<ConceptDecl>(D))
    return HighlightingKind::Concept;
  return llvm::None;
}
llvm::Optional<HighlightingKind> kindForType(const Type *TP) {
  if (!TP)
    return llvm::None;
  if (TP->isBuiltinType()) // Builtins are special, they do not have decls.
    return HighlightingKind::Primitive;
  if (auto *TD = dyn_cast<TemplateTypeParmType>(TP))
    return kindForDecl(TD->getDecl());
  if (auto *TD = TP->getAsTagDecl())
    return kindForDecl(TD);
  return llvm::None;
}

llvm::Optional<HighlightingKind> kindForReference(const ReferenceLoc &R) {
  llvm::Optional<HighlightingKind> Result;
  for (const NamedDecl *Decl : R.Targets) {
    if (!canHighlightName(Decl->getDeclName()))
      return llvm::None;
    auto Kind = kindForDecl(Decl);
    if (!Kind || (Result && Kind != Result))
      return llvm::None;
    Result = Kind;
  }
  return Result;
}

// For a macro usage `DUMP(foo)`, we want:
//  - DUMP --> "macro"
//  - foo --> "variable".
SourceLocation getHighlightableSpellingToken(SourceLocation L,
                                             const SourceManager &SM) {
  if (L.isFileID())
    return SM.isWrittenInMainFile(L) ? L : SourceLocation{};
  // Tokens expanded from the macro body contribute no highlightings.
  if (!SM.isMacroArgExpansion(L))
    return {};
  // Tokens expanded from macro args are potentially highlightable.
  return getHighlightableSpellingToken(SM.getImmediateSpellingLoc(L), SM);
}

unsigned evaluateHighlightPriority(HighlightingKind Kind) {
  enum HighlightPriority { Dependent = 0, Resolved = 1 };
  return Kind == HighlightingKind::DependentType ||
                 Kind == HighlightingKind::DependentName
             ? Dependent
             : Resolved;
}

// Sometimes we get conflicts between findExplicitReferences() returning
// a heuristic result for a dependent name (e.g. Method) and
// CollectExtraHighlighting returning a fallback dependent highlighting (e.g.
// DependentName). In such cases, resolve the conflict in favour of the
// resolved (non-dependent) highlighting.
// With macros we can get other conflicts (if a spelled token has multiple
// expansions with different token types) which we can't usefully resolve.
llvm::Optional<HighlightingToken>
resolveConflict(ArrayRef<HighlightingToken> Tokens) {
  if (Tokens.size() == 1)
    return Tokens[0];

  if (Tokens.size() != 2)
    return llvm::None;

  unsigned Priority1 = evaluateHighlightPriority(Tokens[0].Kind);
  unsigned Priority2 = evaluateHighlightPriority(Tokens[1].Kind);
  if (Priority1 == Priority2)
    return llvm::None;
  return Priority1 > Priority2 ? Tokens[0] : Tokens[1];
}

/// Consumes source locations and maps them to text ranges for highlightings.
class HighlightingsBuilder {
public:
  HighlightingsBuilder(const ParsedAST &AST)
      : TB(AST.getTokens()), SourceMgr(AST.getSourceManager()),
        LangOpts(AST.getLangOpts()) {}

  void addToken(HighlightingToken T) { Tokens.push_back(T); }

  void addToken(SourceLocation Loc, HighlightingKind Kind) {
    Loc = getHighlightableSpellingToken(Loc, SourceMgr);
    if (Loc.isInvalid())
      return;
    const auto *Tok = TB.spelledTokenAt(Loc);
    assert(Tok);

    auto Range = halfOpenToRange(SourceMgr,
                                 Tok->range(SourceMgr).toCharRange(SourceMgr));
    Tokens.push_back(HighlightingToken{Kind, std::move(Range)});
  }

  std::vector<HighlightingToken> collect(ParsedAST &AST) && {
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
      if (auto Resolved = resolveConflict(Conflicting))
        NonConflicting.push_back(*Resolved);
      // TokRef[Conflicting.size()] is the next token with a different range (or
      // the end of the Tokens).
      TokRef = TokRef.drop_front(Conflicting.size());
    }
    const auto &SM = AST.getSourceManager();
    StringRef MainCode = SM.getBufferOrFake(SM.getMainFileID()).getBuffer();

    // Merge token stream with "inactive line" markers.
    std::vector<HighlightingToken> WithInactiveLines;
    auto SortedSkippedRanges = AST.getMacros().SkippedRanges;
    llvm::sort(SortedSkippedRanges);
    auto It = NonConflicting.begin();
    for (const Range &R : SortedSkippedRanges) {
      // Create one token for each line in the skipped range, so it works
      // with line-based diffing.
      assert(R.start.line <= R.end.line);
      for (int Line = R.start.line; Line <= R.end.line; ++Line) {
        // If the end of the inactive range is at the beginning
        // of a line, that line is not inactive.
        if (Line == R.end.line && R.end.character == 0)
          continue;
        // Copy tokens before the inactive line
        for (; It != NonConflicting.end() && It->R.start.line < Line; ++It)
          WithInactiveLines.push_back(std::move(*It));
        // Add a token for the inactive line itself.
        auto StartOfLine = positionToOffset(MainCode, Position{Line, 0});
        if (StartOfLine) {
          StringRef LineText =
              MainCode.drop_front(*StartOfLine).take_until([](char C) {
                return C == '\n';
              });
          WithInactiveLines.push_back(
              {HighlightingKind::InactiveCode,
               {Position{Line, 0},
                Position{Line, static_cast<int>(lspLength(LineText))}}});
        } else {
          elog("Failed to convert position to offset: {0}",
               StartOfLine.takeError());
        }

        // Skip any other tokens on the inactive line. e.g.
        // `#ifndef Foo` is considered as part of an inactive region when Foo is
        // defined, and there is a Foo macro token.
        // FIXME: we should reduce the scope of the inactive region to not
        // include the directive itself.
        while (It != NonConflicting.end() && It->R.start.line == Line)
          ++It;
      }
    }
    // Copy tokens after the last inactive line
    for (; It != NonConflicting.end(); ++It)
      WithInactiveLines.push_back(std::move(*It));
    return WithInactiveLines;
  }

private:
  const syntax::TokenBuffer &TB;
  const SourceManager &SourceMgr;
  const LangOptions &LangOpts;
  std::vector<HighlightingToken> Tokens;
};

/// Produces highlightings, which are not captured by findExplicitReferences,
/// e.g. highlights dependent names and 'auto' as the underlying type.
class CollectExtraHighlightings
    : public RecursiveASTVisitor<CollectExtraHighlightings> {
public:
  CollectExtraHighlightings(HighlightingsBuilder &H) : H(H) {}

  bool VisitDecltypeTypeLoc(DecltypeTypeLoc L) {
    if (auto K = kindForType(L.getTypePtr()))
      H.addToken(L.getBeginLoc(), *K);
    return true;
  }

  bool VisitDeclaratorDecl(DeclaratorDecl *D) {
    auto *AT = D->getType()->getContainedAutoType();
    if (!AT)
      return true;
    if (auto K = kindForType(AT->getDeducedType().getTypePtrOrNull()))
      H.addToken(D->getTypeSpecStartLoc(), *K);
    return true;
  }

  bool VisitOverloadExpr(OverloadExpr *E) {
    if (!E->decls().empty())
      return true; // handled by findExplicitReferences.
    H.addToken(E->getNameLoc(), HighlightingKind::DependentName);
    return true;
  }

  bool VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr *E) {
    H.addToken(E->getMemberNameInfo().getLoc(),
               HighlightingKind::DependentName);
    return true;
  }

  bool VisitDependentScopeDeclRefExpr(DependentScopeDeclRefExpr *E) {
    H.addToken(E->getNameInfo().getLoc(), HighlightingKind::DependentName);
    return true;
  }

  bool VisitDependentNameTypeLoc(DependentNameTypeLoc L) {
    H.addToken(L.getNameLoc(), HighlightingKind::DependentType);
    return true;
  }

  bool VisitDependentTemplateSpecializationTypeLoc(
      DependentTemplateSpecializationTypeLoc L) {
    H.addToken(L.getTemplateNameLoc(), HighlightingKind::DependentType);
    return true;
  }

  bool TraverseTemplateArgumentLoc(TemplateArgumentLoc L) {
    switch (L.getArgument().getKind()) {
    case TemplateArgument::Template:
    case TemplateArgument::TemplateExpansion:
      H.addToken(L.getTemplateNameLoc(), HighlightingKind::DependentType);
      break;
    default:
      break;
    }
    return RecursiveASTVisitor::TraverseTemplateArgumentLoc(L);
  }

  // findExplicitReferences will walk nested-name-specifiers and
  // find anything that can be resolved to a Decl. However, non-leaf
  // components of nested-name-specifiers which are dependent names
  // (kind "Identifier") cannot be resolved to a decl, so we visit
  // them here.
  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc Q) {
    if (NestedNameSpecifier *NNS = Q.getNestedNameSpecifier()) {
      if (NNS->getKind() == NestedNameSpecifier::Identifier)
        H.addToken(Q.getLocalBeginLoc(), HighlightingKind::DependentType);
    }
    return RecursiveASTVisitor::TraverseNestedNameSpecifierLoc(Q);
  }

private:
  HighlightingsBuilder &H;
};

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

std::vector<HighlightingToken> getSemanticHighlightings(ParsedAST &AST) {
  auto &C = AST.getASTContext();
  // Add highlightings for AST nodes.
  HighlightingsBuilder Builder(AST);
  // Highlight 'decltype' and 'auto' as their underlying types.
  CollectExtraHighlightings(Builder).TraverseAST(C);
  // Highlight all decls and references coming from the AST.
  findExplicitReferences(C, [&](ReferenceLoc R) {
    if (auto Kind = kindForReference(R))
      Builder.addToken(R.NameLoc, *Kind);
  });
  // Add highlightings for macro references.
  for (const auto &SIDToRefs : AST.getMacros().MacroRefs) {
    for (const auto &M : SIDToRefs.second)
      Builder.addToken({HighlightingKind::Macro, M});
  }
  for (const auto &M : AST.getMacros().UnknownMacros)
    Builder.addToken({HighlightingKind::Macro, M});

  return std::move(Builder).collect(AST);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, HighlightingKind K) {
  switch (K) {
  case HighlightingKind::Variable:
    return OS << "Variable";
  case HighlightingKind::LocalVariable:
    return OS << "LocalVariable";
  case HighlightingKind::Parameter:
    return OS << "Parameter";
  case HighlightingKind::Function:
    return OS << "Function";
  case HighlightingKind::Method:
    return OS << "Method";
  case HighlightingKind::StaticMethod:
    return OS << "StaticMethod";
  case HighlightingKind::Field:
    return OS << "Field";
  case HighlightingKind::StaticField:
    return OS << "StaticField";
  case HighlightingKind::Class:
    return OS << "Class";
  case HighlightingKind::Enum:
    return OS << "Enum";
  case HighlightingKind::EnumConstant:
    return OS << "EnumConstant";
  case HighlightingKind::Typedef:
    return OS << "Typedef";
  case HighlightingKind::DependentType:
    return OS << "DependentType";
  case HighlightingKind::DependentName:
    return OS << "DependentName";
  case HighlightingKind::Namespace:
    return OS << "Namespace";
  case HighlightingKind::TemplateParameter:
    return OS << "TemplateParameter";
  case HighlightingKind::Concept:
    return OS << "Concept";
  case HighlightingKind::Primitive:
    return OS << "Primitive";
  case HighlightingKind::Macro:
    return OS << "Macro";
  case HighlightingKind::InactiveCode:
    return OS << "InactiveCode";
  }
  llvm_unreachable("invalid HighlightingKind");
}

std::vector<LineHighlightings>
diffHighlightings(ArrayRef<HighlightingToken> New,
                  ArrayRef<HighlightingToken> Old) {
  assert(std::is_sorted(New.begin(), New.end()) &&
         "New must be a sorted vector");
  assert(std::is_sorted(Old.begin(), Old.end()) &&
         "Old must be a sorted vector");

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

  for (int LineNumber = 0; NewLine.end() < NewEnd || OldLine.end() < OldEnd;
       LineNumber = NextLineNumber()) {
    NewLine = takeLine(New, NewLine.end(), LineNumber);
    OldLine = takeLine(Old, OldLine.end(), LineNumber);
    if (NewLine != OldLine) {
      DiffedLines.push_back({LineNumber, NewLine, /*IsInactive=*/false});

      // Turn a HighlightingKind::InactiveCode token into the IsInactive flag.
      auto &AddedLine = DiffedLines.back();
      llvm::erase_if(AddedLine.Tokens, [&](const HighlightingToken &T) {
        if (T.Kind == HighlightingKind::InactiveCode) {
          AddedLine.IsInactive = true;
          return true;
        }
        return false;
      });
    }
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

std::vector<SemanticToken>
toSemanticTokens(llvm::ArrayRef<HighlightingToken> Tokens) {
  assert(std::is_sorted(Tokens.begin(), Tokens.end()));
  std::vector<SemanticToken> Result;
  const HighlightingToken *Last = nullptr;
  for (const HighlightingToken &Tok : Tokens) {
    Result.emplace_back();
    SemanticToken &Out = Result.back();
    // deltaStart/deltaLine are relative if possible.
    if (Last) {
      assert(Tok.R.start.line >= Last->R.start.line);
      Out.deltaLine = Tok.R.start.line - Last->R.start.line;
      if (Out.deltaLine == 0) {
        assert(Tok.R.start.character >= Last->R.start.character);
        Out.deltaStart = Tok.R.start.character - Last->R.start.character;
      } else {
        Out.deltaStart = Tok.R.start.character;
      }
    } else {
      Out.deltaLine = Tok.R.start.line;
      Out.deltaStart = Tok.R.start.character;
    }
    assert(Tok.R.end.line == Tok.R.start.line);
    Out.length = Tok.R.end.character - Tok.R.start.character;
    Out.tokenType = static_cast<unsigned>(Tok.Kind);

    Last = &Tok;
  }
  return Result;
}
llvm::StringRef toSemanticTokenType(HighlightingKind Kind) {
  switch (Kind) {
  case HighlightingKind::Variable:
  case HighlightingKind::LocalVariable:
  case HighlightingKind::StaticField:
    return "variable";
  case HighlightingKind::Parameter:
    return "parameter";
  case HighlightingKind::Function:
    return "function";
  case HighlightingKind::Method:
    return "method";
  case HighlightingKind::StaticMethod:
    // FIXME: better method with static modifier?
    return "function";
  case HighlightingKind::Field:
    return "property";
  case HighlightingKind::Class:
    return "class";
  case HighlightingKind::Enum:
    return "enum";
  case HighlightingKind::EnumConstant:
    return "enumConstant"; // nonstandard
  case HighlightingKind::Typedef:
    return "type";
  case HighlightingKind::DependentType:
    return "dependent"; // nonstandard
  case HighlightingKind::DependentName:
    return "dependent"; // nonstandard
  case HighlightingKind::Namespace:
    return "namespace";
  case HighlightingKind::TemplateParameter:
    return "typeParameter";
  case HighlightingKind::Concept:
    return "concept"; // nonstandard
  case HighlightingKind::Primitive:
    return "type";
  case HighlightingKind::Macro:
    return "macro";
  case HighlightingKind::InactiveCode:
    return "comment";
  }
  llvm_unreachable("unhandled HighlightingKind");
}

std::vector<TheiaSemanticHighlightingInformation>
toTheiaSemanticHighlightingInformation(
    llvm::ArrayRef<LineHighlightings> Tokens) {
  if (Tokens.size() == 0)
    return {};

  // FIXME: Tokens might be multiple lines long (block comments) in this case
  // this needs to add multiple lines for those tokens.
  std::vector<TheiaSemanticHighlightingInformation> Lines;
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

    Lines.push_back({Line.Line, encodeBase64(LineByteTokens), Line.IsInactive});
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
  case HighlightingKind::StaticMethod:
    return "entity.name.function.method.static.cpp";
  case HighlightingKind::Variable:
    return "variable.other.cpp";
  case HighlightingKind::LocalVariable:
    return "variable.other.local.cpp";
  case HighlightingKind::Parameter:
    return "variable.parameter.cpp";
  case HighlightingKind::Field:
    return "variable.other.field.cpp";
  case HighlightingKind::StaticField:
    return "variable.other.field.static.cpp";
  case HighlightingKind::Class:
    return "entity.name.type.class.cpp";
  case HighlightingKind::Enum:
    return "entity.name.type.enum.cpp";
  case HighlightingKind::EnumConstant:
    return "variable.other.enummember.cpp";
  case HighlightingKind::Typedef:
    return "entity.name.type.typedef.cpp";
  case HighlightingKind::DependentType:
    return "entity.name.type.dependent.cpp";
  case HighlightingKind::DependentName:
    return "entity.name.other.dependent.cpp";
  case HighlightingKind::Namespace:
    return "entity.name.namespace.cpp";
  case HighlightingKind::TemplateParameter:
    return "entity.name.type.template.cpp";
  case HighlightingKind::Concept:
    return "entity.name.type.concept.cpp";
  case HighlightingKind::Primitive:
    return "storage.type.primitive.cpp";
  case HighlightingKind::Macro:
    return "entity.name.function.preprocessor.cpp";
  case HighlightingKind::InactiveCode:
    return "meta.disabled";
  }
  llvm_unreachable("unhandled HighlightingKind");
}

std::vector<SemanticTokensEdit>
diffTokens(llvm::ArrayRef<SemanticToken> Old,
           llvm::ArrayRef<SemanticToken> New) {
  // For now, just replace everything from the first-last modification.
  // FIXME: use a real diff instead, this is bad with include-insertion.

  unsigned Offset = 0;
  while (!Old.empty() && !New.empty() && Old.front() == New.front()) {
    ++Offset;
    Old = Old.drop_front();
    New = New.drop_front();
  }
  while (!Old.empty() && !New.empty() && Old.back() == New.back()) {
    Old = Old.drop_back();
    New = New.drop_back();
  }

  if (Old.empty() && New.empty())
    return {};
  SemanticTokensEdit Edit;
  Edit.startToken = Offset;
  Edit.deleteTokens = Old.size();
  Edit.tokens = New;
  return {std::move(Edit)};
}

} // namespace clangd
} // namespace clang
