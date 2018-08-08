//===--- Quality.cpp --------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
#include "Quality.h"
#include "FileDistance.h"
#include "URI.h"
#include "index/Index.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>

namespace clang {
namespace clangd {
using namespace llvm;
static bool isReserved(StringRef Name) {
  // FIXME: Should we exclude _Bool and others recognized by the standard?
  return Name.size() >= 2 && Name[0] == '_' &&
         (isUppercase(Name[1]) || Name[1] == '_');
}

static bool hasDeclInMainFile(const Decl &D) {
  auto &SourceMgr = D.getASTContext().getSourceManager();
  for (auto *Redecl : D.redecls()) {
    auto Loc = SourceMgr.getSpellingLoc(Redecl->getLocation());
    if (SourceMgr.isWrittenInMainFile(Loc))
      return true;
  }
  return false;
}

static bool hasUsingDeclInMainFile(const CodeCompletionResult &R) {
  const auto &Context = R.Declaration->getASTContext();
  const auto &SourceMgr = Context.getSourceManager();
  if (R.ShadowDecl) {
    const auto Loc = SourceMgr.getExpansionLoc(R.ShadowDecl->getLocation());
    if (SourceMgr.isWrittenInMainFile(Loc))
      return true;
  }
  return false;
}

static SymbolQualitySignals::SymbolCategory categorize(const NamedDecl &ND) {
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
    return SymbolQualitySignals::Type;
  case index::SymbolKind::Function:
  case index::SymbolKind::ClassMethod:
  case index::SymbolKind::InstanceMethod:
  case index::SymbolKind::StaticMethod:
  case index::SymbolKind::InstanceProperty:
  case index::SymbolKind::ClassProperty:
  case index::SymbolKind::StaticProperty:
  case index::SymbolKind::Destructor:
  case index::SymbolKind::ConversionFunction:
    return SymbolQualitySignals::Function;
  case index::SymbolKind::Constructor:
    return SymbolQualitySignals::Constructor;
  case index::SymbolKind::Variable:
  case index::SymbolKind::Field:
  case index::SymbolKind::EnumConstant:
  case index::SymbolKind::Parameter:
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
  if (SemaCCResult.Availability == CXAvailability_Deprecated)
    Deprecated = true;

  Category = categorize(SemaCCResult);

  if (SemaCCResult.Declaration) {
    if (auto *ID = SemaCCResult.Declaration->getIdentifier())
      ReservedName = ReservedName || isReserved(ID->getName());
  } else if (SemaCCResult.Kind == CodeCompletionResult::RK_Macro)
    ReservedName = ReservedName || isReserved(SemaCCResult.Macro->getName());
}

void SymbolQualitySignals::merge(const Symbol &IndexResult) {
  References = std::max(IndexResult.References, References);
  Category = categorize(IndexResult.SymInfo);
  ReservedName = ReservedName || isReserved(IndexResult.Name);
}

float SymbolQualitySignals::evaluate() const {
  float Score = 1;

  // This avoids a sharp gradient for tail symbols, and also neatly avoids the
  // question of whether 0 references means a bad symbol or missing data.
  if (References >= 10) {
    // Use a sigmoid style boosting function, which flats out nicely for large
    // numbers (e.g. 2.58 for 1M refererences).
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
    Score *= 0.2f;
    break;
  case Unknown:
  case Constructor: // No boost constructors so they are after class types.
    break;
  }

  return Score;
}

raw_ostream &operator<<(raw_ostream &OS, const SymbolQualitySignals &S) {
  OS << formatv("=== Symbol quality: {0}\n", S.evaluate());
  OS << formatv("\tReferences: {0}\n", S.References);
  OS << formatv("\tDeprecated: {0}\n", S.Deprecated);
  OS << formatv("\tReserved name: {0}\n", S.ReservedName);
  OS << formatv("\tCategory: {0}\n", static_cast<int>(S.Category));
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
  // This threshold could be tweaked, e.g. to treat module-visible as global.
  if (D->getLinkageInternal() < ExternalLinkage)
    return SymbolRelevanceSignals::FileScope;
  return SymbolRelevanceSignals::GlobalScope;
}

void SymbolRelevanceSignals::merge(const Symbol &IndexResult) {
  // FIXME: Index results always assumed to be at global scope. If Scope becomes
  // relevant to non-completion requests, we should recognize class members etc.

  SymbolURI = IndexResult.CanonicalDeclaration.FileURI;
  IsInstanceMember |= isInstanceMember(IndexResult.SymInfo);
}

void SymbolRelevanceSignals::merge(const CodeCompletionResult &SemaCCResult) {
  if (SemaCCResult.Availability == CXAvailability_NotAvailable ||
      SemaCCResult.Availability == CXAvailability_NotAccessible)
    Forbidden = true;

  if (SemaCCResult.Declaration) {
    // We boost things that have decls in the main file. We give a fixed score
    // for all other declarations in sema as they are already included in the
    // translation unit.
    float DeclProximity = (hasDeclInMainFile(*SemaCCResult.Declaration) ||
                           hasUsingDeclInMainFile(SemaCCResult))
                              ? 1.0
                              : 0.6;
    SemaProximityScore = std::max(DeclProximity, SemaProximityScore);
    IsInstanceMember |= isInstanceMember(SemaCCResult.Declaration);
  }

  // Declarations are scoped, others (like macros) are assumed global.
  if (SemaCCResult.Declaration)
    Scope = std::min(Scope, computeScope(SemaCCResult.Declaration));

  NeedsFixIts = !SemaCCResult.FixIts.empty();
}

static std::pair<float, unsigned> proximityScore(llvm::StringRef SymbolURI,
                                                 URIDistance *D) {
  if (!D || SymbolURI.empty())
    return {0.f, 0u};
  unsigned Distance = D->distance(SymbolURI);
  // Assume approximately default options are used for sensible scoring.
  return {std::exp(Distance * -0.4f / FileDistanceOptions().UpCost), Distance};
}

float SymbolRelevanceSignals::evaluate() const {
  float Score = 1;

  if (Forbidden)
    return 0;

  Score *= NameMatch;

  // Proximity scores are [0,1] and we translate them into a multiplier in the
  // range from 1 to 3.
  Score *= 1 + 2 * std::max(proximityScore(SymbolURI, FileProximityMatch).first,
                            SemaProximityScore);

  // Symbols like local variables may only be referenced within their scope.
  // Conversely if we're in that scope, it's likely we'll reference them.
  if (Query == CodeComplete) {
    // The narrower the scope where a symbol is visible, the more likely it is
    // to be relevant when it is available.
    switch (Scope) {
    case GlobalScope:
      break;
    case FileScope:
      Score *= 1.5;
      break;
    case ClassScope:
      Score *= 2;
      break;
    case FunctionScope:
      Score *= 4;
      break;
    }
  }

  // Penalize non-instance members when they are accessed via a class instance.
  if (!IsInstanceMember &&
      (Context == CodeCompletionContext::CCC_DotMemberAccess ||
       Context == CodeCompletionContext::CCC_ArrowMemberAccess)) {
    Score *= 0.5;
  }

  // Penalize for FixIts.
  if (NeedsFixIts)
    Score *= 0.5;

  return Score;
}

raw_ostream &operator<<(raw_ostream &OS, const SymbolRelevanceSignals &S) {
  OS << formatv("=== Symbol relevance: {0}\n", S.evaluate());
  OS << formatv("\tName match: {0}\n", S.NameMatch);
  OS << formatv("\tForbidden: {0}\n", S.Forbidden);
  OS << formatv("\tNeedsFixIts: {0}\n", S.NeedsFixIts);
  OS << formatv("\tIsInstanceMember: {0}\n", S.IsInstanceMember);
  OS << formatv("\tContext: {0}\n", getCompletionKindString(S.Context));
  OS << formatv("\tSymbol URI: {0}\n", S.SymbolURI);
  if (S.FileProximityMatch) {
    auto Score = proximityScore(S.SymbolURI, S.FileProximityMatch);
    OS << formatv("\tIndex proximity: {0} (distance={1})\n", Score.first,
                  Score.second);
  }
  OS << formatv("\tSema proximity: {0}\n", S.SemaProximityScore);
  OS << formatv("\tQuery type: {0}\n", static_cast<int>(S.Query));
  OS << formatv("\tScope: {0}\n", static_cast<int>(S.Scope));
  return OS;
}

float evaluateSymbolAndRelevance(float SymbolQuality, float SymbolRelevance) {
  return SymbolQuality * SymbolRelevance;
}

// Produces an integer that sorts in the same order as F.
// That is: a < b <==> encodeFloat(a) < encodeFloat(b).
static uint32_t encodeFloat(float F) {
  static_assert(std::numeric_limits<float>::is_iec559, "");
  constexpr uint32_t TopBit = ~(~uint32_t{0} >> 1);

  // Get the bits of the float. Endianness is the same as for integers.
  uint32_t U = FloatToBits(F);
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
  write_hex(OS, encodeFloat(-Score), llvm::HexPrintStyle::Lower,
            /*Width=*/2 * sizeof(Score));
  OS << Name;
  OS.flush();
  return S;
}

} // namespace clangd
} // namespace clang
