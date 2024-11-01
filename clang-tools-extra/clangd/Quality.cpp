//===--- Quality.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Quality.h"
#include "AST.h"
#include "ASTSignals.h"
#include "FileDistance.h"
#include "SourceCode.h"
#include "index/Symbol.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>

namespace clang {
namespace clangd {

static bool hasDeclInMainFile(const Decl &D) {
  auto &SourceMgr = D.getASTContext().getSourceManager();
  for (auto *Redecl : D.redecls()) {
    if (isInsideMainFile(Redecl->getLocation(), SourceMgr))
      return true;
  }
  return false;
}

static bool hasUsingDeclInMainFile(const CodeCompletionResult &R) {
  const auto &Context = R.Declaration->getASTContext();
  const auto &SourceMgr = Context.getSourceManager();
  if (R.ShadowDecl) {
    if (isInsideMainFile(R.ShadowDecl->getLocation(), SourceMgr))
      return true;
  }
  return false;
}

static SymbolQualitySignals::SymbolCategory categorize(const NamedDecl &ND) {
  if (const auto *FD = dyn_cast<FunctionDecl>(&ND)) {
    if (FD->isOverloadedOperator())
      return SymbolQualitySignals::Operator;
  }
  class Switch
      : public ConstDeclVisitor<Switch, SymbolQualitySignals::SymbolCategory> {
  public:
#define MAP(DeclType, Category)                                                \
  SymbolQualitySignals::SymbolCategory Visit##DeclType(const DeclType *) {     \
    return SymbolQualitySignals::Category;                                     \
  }
    MAP(NamespaceDecl, Namespace);
    MAP(NamespaceAliasDecl, Namespace);
    MAP(TypeDecl, Type);
    MAP(TypeAliasTemplateDecl, Type);
    MAP(ClassTemplateDecl, Type);
    MAP(CXXConstructorDecl, Constructor);
    MAP(CXXDestructorDecl, Destructor);
    MAP(ValueDecl, Variable);
    MAP(VarTemplateDecl, Variable);
    MAP(FunctionDecl, Function);
    MAP(FunctionTemplateDecl, Function);
    MAP(Decl, Unknown);
#undef MAP
  };
  return Switch().Visit(&ND);
}

static SymbolQualitySignals::SymbolCategory
categorize(const CodeCompletionResult &R) {
  if (R.Declaration)
    return categorize(*R.Declaration);
  if (R.Kind == CodeCompletionResult::RK_Macro)
    return SymbolQualitySignals::Macro;
  // Everything else is a keyword or a pattern. Patterns are mostly keywords
  // too, except a few which we recognize by cursor kind.
  switch (R.CursorKind) {
  case CXCursor_CXXMethod:
    return SymbolQualitySignals::Function;
  case CXCursor_ModuleImportDecl:
    return SymbolQualitySignals::Namespace;
  case CXCursor_MacroDefinition:
    return SymbolQualitySignals::Macro;
  case CXCursor_TypeRef:
    return SymbolQualitySignals::Type;
  case CXCursor_MemberRef:
    return SymbolQualitySignals::Variable;
  case CXCursor_Constructor:
    return SymbolQualitySignals::Constructor;
  default:
    return SymbolQualitySignals::Keyword;
  }
}

static SymbolQualitySignals::SymbolCategory
categorize(const index::SymbolInfo &D) {
  switch (D.Kind) {
  case index::SymbolKind::Namespace:
  case index::SymbolKind::NamespaceAlias:
    return SymbolQualitySignals::Namespace;
  case index::SymbolKind::Macro:
    return SymbolQualitySignals::Macro;
  case index::SymbolKind::Enum:
  case index::SymbolKind::Struct:
  case index::SymbolKind::Class:
  case index::SymbolKind::Protocol:
  case index::SymbolKind::Extension:
  case index::SymbolKind::Union:
  case index::SymbolKind::TypeAlias:
  case index::SymbolKind::TemplateTypeParm:
  case index::SymbolKind::TemplateTemplateParm:
  case index::SymbolKind::Concept:
    return SymbolQualitySignals::Type;
  case index::SymbolKind::Function:
  case index::SymbolKind::ClassMethod:
  case index::SymbolKind::InstanceMethod:
  case index::SymbolKind::StaticMethod:
  case index::SymbolKind::InstanceProperty:
  case index::SymbolKind::ClassProperty:
  case index::SymbolKind::StaticProperty:
  case index::SymbolKind::ConversionFunction:
    return SymbolQualitySignals::Function;
  case index::SymbolKind::Destructor:
    return SymbolQualitySignals::Destructor;
  case index::SymbolKind::Constructor:
    return SymbolQualitySignals::Constructor;
  case index::SymbolKind::Variable:
  case index::SymbolKind::Field:
  case index::SymbolKind::EnumConstant:
  case index::SymbolKind::Parameter:
  case index::SymbolKind::NonTypeTemplateParm:
    return SymbolQualitySignals::Variable;
  case index::SymbolKind::Using:
  case index::SymbolKind::Module:
  case index::SymbolKind::Unknown:
    return SymbolQualitySignals::Unknown;
  }
  llvm_unreachable("Unknown index::SymbolKind");
}

static bool isInstanceMember(const NamedDecl *ND) {
  if (!ND)
    return false;
  if (const auto *TP = dyn_cast<FunctionTemplateDecl>(ND))
    ND = TP->TemplateDecl::getTemplatedDecl();
  if (const auto *CM = dyn_cast<CXXMethodDecl>(ND))
    return !CM->isStatic();
  return isa<FieldDecl>(ND); // Note that static fields are VarDecl.
}

static bool isInstanceMember(const index::SymbolInfo &D) {
  switch (D.Kind) {
  case index::SymbolKind::InstanceMethod:
  case index::SymbolKind::InstanceProperty:
  case index::SymbolKind::Field:
    return true;
  default:
    return false;
  }
}

void SymbolQualitySignals::merge(const CodeCompletionResult &SemaCCResult) {
  Deprecated |= (SemaCCResult.Availability == CXAvailability_Deprecated);
  Category = categorize(SemaCCResult);

  if (SemaCCResult.Declaration) {
    ImplementationDetail |= isImplementationDetail(SemaCCResult.Declaration);
    if (auto *ID = SemaCCResult.Declaration->getIdentifier())
      ReservedName = ReservedName || isReservedName(ID->getName());
  } else if (SemaCCResult.Kind == CodeCompletionResult::RK_Macro)
    ReservedName =
        ReservedName || isReservedName(SemaCCResult.Macro->getName());
}

void SymbolQualitySignals::merge(const Symbol &IndexResult) {
  Deprecated |= (IndexResult.Flags & Symbol::Deprecated);
  ImplementationDetail |= (IndexResult.Flags & Symbol::ImplementationDetail);
  References = std::max(IndexResult.References, References);
  Category = categorize(IndexResult.SymInfo);
  ReservedName = ReservedName || isReservedName(IndexResult.Name);
}

float SymbolQualitySignals::evaluateHeuristics() const {
  float Score = 1;

  // This avoids a sharp gradient for tail symbols, and also neatly avoids the
  // question of whether 0 references means a bad symbol or missing data.
  if (References >= 10) {
    // Use a sigmoid style boosting function, which flats out nicely for large
    // numbers (e.g. 2.58 for 1M references).
    // The following boosting function is equivalent to:
    //   m = 0.06
    //   f = 12.0
    //   boost = f * sigmoid(m * std::log(References)) - 0.5 * f + 0.59
    // Sample data points: (10, 1.00), (100, 1.41), (1000, 1.82),
    //                     (10K, 2.21), (100K, 2.58), (1M, 2.94)
    float S = std::pow(References, -0.06);
    Score *= 6.0 * (1 - S) / (1 + S) + 0.59;
  }

  if (Deprecated)
    Score *= 0.1f;
  if (ReservedName)
    Score *= 0.1f;
  if (ImplementationDetail)
    Score *= 0.2f;

  switch (Category) {
  case Keyword: // Often relevant, but misses most signals.
    Score *= 4; // FIXME: important keywords should have specific boosts.
    break;
  case Type:
  case Function:
  case Variable:
    Score *= 1.1f;
    break;
  case Namespace:
    Score *= 0.8f;
    break;
  case Macro:
  case Destructor:
  case Operator:
    Score *= 0.5f;
    break;
  case Constructor: // No boost constructors so they are after class types.
  case Unknown:
    break;
  }

  return Score;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const SymbolQualitySignals &S) {
  OS << llvm::formatv("=== Symbol quality: {0}\n", S.evaluateHeuristics());
  OS << llvm::formatv("\tReferences: {0}\n", S.References);
  OS << llvm::formatv("\tDeprecated: {0}\n", S.Deprecated);
  OS << llvm::formatv("\tReserved name: {0}\n", S.ReservedName);
  OS << llvm::formatv("\tImplementation detail: {0}\n", S.ImplementationDetail);
  OS << llvm::formatv("\tCategory: {0}\n", static_cast<int>(S.Category));
  return OS;
}

static SymbolRelevanceSignals::AccessibleScope
computeScope(const NamedDecl *D) {
  // Injected "Foo" within the class "Foo" has file scope, not class scope.
  const DeclContext *DC = D->getDeclContext();
  if (auto *R = dyn_cast_or_null<RecordDecl>(D))
    if (R->isInjectedClassName())
      DC = DC->getParent();
  // Class constructor should have the same scope as the class.
  if (isa<CXXConstructorDecl>(D))
    DC = DC->getParent();
  bool InClass = false;
  for (; !DC->isFileContext(); DC = DC->getParent()) {
    if (DC->isFunctionOrMethod())
      return SymbolRelevanceSignals::FunctionScope;
    InClass = InClass || DC->isRecord();
  }
  if (InClass)
    return SymbolRelevanceSignals::ClassScope;
  // ExternalLinkage threshold could be tweaked, e.g. module-visible as global.
  // Avoid caching linkage if it may change after enclosing code completion.
  if (hasUnstableLinkage(D) || D->getLinkageInternal() < ExternalLinkage)
    return SymbolRelevanceSignals::FileScope;
  return SymbolRelevanceSignals::GlobalScope;
}

void SymbolRelevanceSignals::merge(const Symbol &IndexResult) {
  SymbolURI = IndexResult.CanonicalDeclaration.FileURI;
  SymbolScope = IndexResult.Scope;
  IsInstanceMember |= isInstanceMember(IndexResult.SymInfo);
  if (!(IndexResult.Flags & Symbol::VisibleOutsideFile)) {
    Scope = AccessibleScope::FileScope;
  }
  if (MainFileSignals) {
    MainFileRefs =
        std::max(MainFileRefs,
                 MainFileSignals->ReferencedSymbols.lookup(IndexResult.ID));
    ScopeRefsInFile =
        std::max(ScopeRefsInFile,
                 MainFileSignals->RelatedNamespaces.lookup(IndexResult.Scope));
  }
}

void SymbolRelevanceSignals::computeASTSignals(
    const CodeCompletionResult &SemaResult) {
  if (!MainFileSignals)
    return;
  if ((SemaResult.Kind != CodeCompletionResult::RK_Declaration) &&
      (SemaResult.Kind != CodeCompletionResult::RK_Pattern))
    return;
  if (const NamedDecl *ND = SemaResult.getDeclaration()) {
    auto ID = getSymbolID(ND);
    if (!ID)
      return;
    MainFileRefs =
        std::max(MainFileRefs, MainFileSignals->ReferencedSymbols.lookup(ID));
    if (const auto *NSD = dyn_cast<NamespaceDecl>(ND->getDeclContext())) {
      if (NSD->isAnonymousNamespace())
        return;
      std::string Scope = printNamespaceScope(*NSD);
      if (!Scope.empty())
        ScopeRefsInFile = std::max(
            ScopeRefsInFile, MainFileSignals->RelatedNamespaces.lookup(Scope));
    }
  }
}

void SymbolRelevanceSignals::merge(const CodeCompletionResult &SemaCCResult) {
  if (SemaCCResult.Availability == CXAvailability_NotAvailable ||
      SemaCCResult.Availability == CXAvailability_NotAccessible)
    Forbidden = true;

  if (SemaCCResult.Declaration) {
    SemaSaysInScope = true;
    // We boost things that have decls in the main file. We give a fixed score
    // for all other declarations in sema as they are already included in the
    // translation unit.
    float DeclProximity = (hasDeclInMainFile(*SemaCCResult.Declaration) ||
                           hasUsingDeclInMainFile(SemaCCResult))
                              ? 1.0
                              : 0.6;
    SemaFileProximityScore = std::max(DeclProximity, SemaFileProximityScore);
    IsInstanceMember |= isInstanceMember(SemaCCResult.Declaration);
    InBaseClass |= SemaCCResult.InBaseClass;
  }

  computeASTSignals(SemaCCResult);
  // Declarations are scoped, others (like macros) are assumed global.
  if (SemaCCResult.Declaration)
    Scope = std::min(Scope, computeScope(SemaCCResult.Declaration));

  NeedsFixIts = !SemaCCResult.FixIts.empty();
}

static float fileProximityScore(unsigned FileDistance) {
  // Range: [0, 1]
  // FileDistance = [0, 1, 2, 3, 4, .., FileDistance::Unreachable]
  // Score = [1, 0.82, 0.67, 0.55, 0.45, .., 0]
  if (FileDistance == FileDistance::Unreachable)
    return 0;
  // Assume approximately default options are used for sensible scoring.
  return std::exp(FileDistance * -0.4f / FileDistanceOptions().UpCost);
}

static float scopeProximityScore(unsigned ScopeDistance) {
  // Range: [0.6, 2].
  // ScopeDistance = [0, 1, 2, 3, 4, 5, 6, 7, .., FileDistance::Unreachable]
  // Score = [2.0, 1.55, 1.2, 0.93, 0.72, 0.65, 0.65, 0.65, .., 0.6]
  if (ScopeDistance == FileDistance::Unreachable)
    return 0.6f;
  return std::max(0.65, 2.0 * std::pow(0.6, ScopeDistance / 2.0));
}

static llvm::Optional<llvm::StringRef>
wordMatching(llvm::StringRef Name, const llvm::StringSet<> *ContextWords) {
  if (ContextWords)
    for (const auto &Word : ContextWords->keys())
      if (Name.contains_insensitive(Word))
        return Word;
  return std::nullopt;
}

SymbolRelevanceSignals::DerivedSignals
SymbolRelevanceSignals::calculateDerivedSignals() const {
  DerivedSignals Derived;
  Derived.NameMatchesContext = wordMatching(Name, ContextWords).has_value();
  Derived.FileProximityDistance = !FileProximityMatch || SymbolURI.empty()
                                      ? FileDistance::Unreachable
                                      : FileProximityMatch->distance(SymbolURI);
  if (ScopeProximityMatch) {
    // For global symbol, the distance is 0.
    Derived.ScopeProximityDistance =
        SymbolScope ? ScopeProximityMatch->distance(*SymbolScope) : 0;
  }
  return Derived;
}

float SymbolRelevanceSignals::evaluateHeuristics() const {
  DerivedSignals Derived = calculateDerivedSignals();
  float Score = 1;

  if (Forbidden)
    return 0;

  Score *= NameMatch;

  // File proximity scores are [0,1] and we translate them into a multiplier in
  // the range from 1 to 3.
  Score *= 1 + 2 * std::max(fileProximityScore(Derived.FileProximityDistance),
                            SemaFileProximityScore);

  if (ScopeProximityMatch)
    // Use a constant scope boost for sema results, as scopes of sema results
    // can be tricky (e.g. class/function scope). Set to the max boost as we
    // don't load top-level symbols from the preamble and sema results are
    // always in the accessible scope.
    Score *= SemaSaysInScope
                 ? 2.0
                 : scopeProximityScore(Derived.ScopeProximityDistance);

  if (Derived.NameMatchesContext)
    Score *= 1.5;

  // Symbols like local variables may only be referenced within their scope.
  // Conversely if we're in that scope, it's likely we'll reference them.
  if (Query == CodeComplete) {
    // The narrower the scope where a symbol is visible, the more likely it is
    // to be relevant when it is available.
    switch (Scope) {
    case GlobalScope:
      break;
    case FileScope:
      Score *= 1.5f;
      break;
    case ClassScope:
      Score *= 2;
      break;
    case FunctionScope:
      Score *= 4;
      break;
    }
  } else {
    // For non-completion queries, the wider the scope where a symbol is
    // visible, the more likely it is to be relevant.
    switch (Scope) {
    case GlobalScope:
      break;
    case FileScope:
      Score *= 0.5f;
      break;
    default:
      // TODO: Handle other scopes as we start to use them for index results.
      break;
    }
  }

  if (TypeMatchesPreferred)
    Score *= 5.0;

  // Penalize non-instance members when they are accessed via a class instance.
  if (!IsInstanceMember &&
      (Context == CodeCompletionContext::CCC_DotMemberAccess ||
       Context == CodeCompletionContext::CCC_ArrowMemberAccess)) {
    Score *= 0.2f;
  }

  if (InBaseClass)
    Score *= 0.5f;

  // Penalize for FixIts.
  if (NeedsFixIts)
    Score *= 0.5f;

  // Use a sigmoid style boosting function similar to `References`, which flats
  // out nicely for large values. This avoids a sharp gradient for heavily
  // referenced symbols. Use smaller gradient for ScopeRefsInFile since ideally
  // MainFileRefs <= ScopeRefsInFile.
  if (MainFileRefs >= 2) {
    // E.g.: (2, 1.12), (9, 2.0), (48, 3.0).
    float S = std::pow(MainFileRefs, -0.11);
    Score *= 11.0 * (1 - S) / (1 + S) + 0.7;
  }
  if (ScopeRefsInFile >= 2) {
    // E.g.: (2, 1.04), (14, 2.0), (109, 3.0), (400, 3.6).
    float S = std::pow(ScopeRefsInFile, -0.10);
    Score *= 10.0 * (1 - S) / (1 + S) + 0.7;
  }

  return Score;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const SymbolRelevanceSignals &S) {
  OS << llvm::formatv("=== Symbol relevance: {0}\n", S.evaluateHeuristics());
  OS << llvm::formatv("\tName: {0}\n", S.Name);
  OS << llvm::formatv("\tName match: {0}\n", S.NameMatch);
  if (S.ContextWords)
    OS << llvm::formatv(
        "\tMatching context word: {0}\n",
        wordMatching(S.Name, S.ContextWords).value_or("<none>"));
  OS << llvm::formatv("\tForbidden: {0}\n", S.Forbidden);
  OS << llvm::formatv("\tNeedsFixIts: {0}\n", S.NeedsFixIts);
  OS << llvm::formatv("\tIsInstanceMember: {0}\n", S.IsInstanceMember);
  OS << llvm::formatv("\tInBaseClass: {0}\n", S.InBaseClass);
  OS << llvm::formatv("\tContext: {0}\n", getCompletionKindString(S.Context));
  OS << llvm::formatv("\tQuery type: {0}\n", static_cast<int>(S.Query));
  OS << llvm::formatv("\tScope: {0}\n", static_cast<int>(S.Scope));

  OS << llvm::formatv("\tSymbol URI: {0}\n", S.SymbolURI);
  OS << llvm::formatv("\tSymbol scope: {0}\n",
                      S.SymbolScope ? *S.SymbolScope : "<None>");

  SymbolRelevanceSignals::DerivedSignals Derived = S.calculateDerivedSignals();
  if (S.FileProximityMatch) {
    unsigned Score = fileProximityScore(Derived.FileProximityDistance);
    OS << llvm::formatv("\tIndex URI proximity: {0} (distance={1})\n", Score,
                        Derived.FileProximityDistance);
  }
  OS << llvm::formatv("\tSema file proximity: {0}\n", S.SemaFileProximityScore);

  OS << llvm::formatv("\tSema says in scope: {0}\n", S.SemaSaysInScope);
  if (S.ScopeProximityMatch)
    OS << llvm::formatv("\tIndex scope boost: {0}\n",
                        scopeProximityScore(Derived.ScopeProximityDistance));

  OS << llvm::formatv(
      "\tType matched preferred: {0} (Context type: {1}, Symbol type: {2}\n",
      S.TypeMatchesPreferred, S.HadContextType, S.HadSymbolType);

  return OS;
}

float evaluateSymbolAndRelevance(float SymbolQuality, float SymbolRelevance) {
  return SymbolQuality * SymbolRelevance;
}

// Produces an integer that sorts in the same order as F.
// That is: a < b <==> encodeFloat(a) < encodeFloat(b).
static uint32_t encodeFloat(float F) {
  static_assert(std::numeric_limits<float>::is_iec559);
  constexpr uint32_t TopBit = ~(~uint32_t{0} >> 1);

  // Get the bits of the float. Endianness is the same as for integers.
  uint32_t U = llvm::FloatToBits(F);
  // IEEE 754 floats compare like sign-magnitude integers.
  if (U & TopBit)    // Negative float.
    return 0 - U;    // Map onto the low half of integers, order reversed.
  return U + TopBit; // Positive floats map onto the high half of integers.
}

std::string sortText(float Score, llvm::StringRef Name) {
  // We convert -Score to an integer, and hex-encode for readability.
  // Example: [0.5, "foo"] -> "41000000foo"
  std::string S;
  llvm::raw_string_ostream OS(S);
  llvm::write_hex(OS, encodeFloat(-Score), llvm::HexPrintStyle::Lower,
                  /*Width=*/2 * sizeof(Score));
  OS << Name;
  OS.flush();
  return S;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const SignatureQualitySignals &S) {
  OS << llvm::formatv("=== Signature Quality:\n");
  OS << llvm::formatv("\tNumber of parameters: {0}\n", S.NumberOfParameters);
  OS << llvm::formatv("\tNumber of optional parameters: {0}\n",
                      S.NumberOfOptionalParameters);
  OS << llvm::formatv("\tKind: {0}\n", S.Kind);
  return OS;
}

} // namespace clangd
} // namespace clang
