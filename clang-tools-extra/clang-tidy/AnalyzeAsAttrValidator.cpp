//===--- AnalyzeAsAttrValidator.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalyzeAsAttrValidator.h"
#include "ClangTidyDiagnosticConsumer.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/Pragma.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FormatVariadic.h"
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

using namespace clang::ast_matchers;

namespace clang::tidy {

/// How to build the header AST (the throw-away AST of the standard library
/// header that the annotated types are supposed to mimic). Both fields come
/// verbatim from the pragma; there are no defaults.
struct HeaderASTConfig {
  /// Passed as -isysroot. Empty means "not configured".
  std::string Sysroot;
  /// Header to include as spelled (`<optional>` verbatim, anything else as a
  /// quoted include). Empty means "not configured".
  std::string Header;
  /// Language standard of the header AST, passed as -std= (e.g. `c++23`).
  /// Empty means "not configured".
  std::string Std;
  /// Malformed pragmas; reported at the end of the TU.
  std::vector<SourceLocation> BadPragmaLocs;

  bool isConfigured() const {
    return !Sysroot.empty() && !Header.empty() && !Std.empty();
  }

  friend bool operator<(const HeaderASTConfig &L, const HeaderASTConfig &R) {
    return std::tie(L.Sysroot, L.Header, L.Std) <
           std::tie(R.Sysroot, R.Header, R.Std);
  }
};

namespace {

/// A parsed `analyze_as_method` string such as `emplace(Args&&...)`.
struct MethodSpec {
  std::string Name; ///< `emplace`, `<ClassName>` (constructor), `operator=`.
  /// Parameters as spelled; empty when only a bare name was given. Fuzzy: the
  /// user may spell only a prefix of the real parameter list.
  llvm::SmallVector<std::string, 2> Params;
};

/// One thing to check against the header AST.
struct AttrQuery {
  bool IsClass = false; ///< analyze_as_class (else analyze_as_method)
  std::string Text;     ///< the attribute's string argument
  /// Qualified name of the class the attribute refers to, e.g. `std::optional`:
  /// the analyze_as_class string itself, or for analyze_as_method the one on
  /// the enclosing class.
  std::string Target;
};

} // namespace

//===----------------------------------------------------------------------===//
// Parsing
//===----------------------------------------------------------------------===//

/// Splits `name[(p1, p2, ...)]`. Returns std::nullopt for syntactic garbage
/// (unbalanced `()`/`<>`, empty name or segment, trailing junk).
static std::optional<MethodSpec> parseMethodSpec(llvm::StringRef Text) {
  Text = Text.trim();
  auto [Name, Rest] = Text.split('(');
  MethodSpec Spec;
  Spec.Name = Name.trim().str();
  if (Spec.Name.empty())
    return std::nullopt;
  if (!Text.contains('('))
    return Spec; // bare name

  llvm::StringRef Params = Rest;
  if (!Params.consume_back(")"))
    return std::nullopt; // trailing junk
  if (Params.trim().empty())
    return Spec; // "name()" - verbose but valid

  // Split on commas that are not nested in () or <>.
  llvm::SmallString<8> Open; // currently open brackets, innermost last
  auto AddParam = [&](llvm::StringRef Param) {
    Param = Param.trim();
    if (!Param.empty())
      Spec.Params.push_back(Param.str());
    return !Param.empty();
  };
  size_t Start = 0;
  for (auto [I, C] : llvm::enumerate(Params)) {
    if (C == '(' || C == '<') {
      Open.push_back(C);
    } else if (C == ')' || C == '>') {
      if (Open.empty() || Open.back() != (C == ')' ? '(' : '<'))
        return std::nullopt;
      Open.pop_back();
    } else if (C == ',' && Open.empty()) {
      if (!AddParam(Params.slice(Start, I)))
        return std::nullopt;
      Start = I + 1;
    }
  }
  if (!Open.empty() || !AddParam(Params.substr(Start)))
    return std::nullopt;
  return Spec;
}

//===----------------------------------------------------------------------===//
// Attribute string -> matcher
//===----------------------------------------------------------------------===//

namespace {

/// Spelling normalisation for fuzzy type comparison: drops namespace
/// qualifiers, `const`, `&`, `&&`, `...` and whitespace, so `std::nullopt_t`,
/// `nullopt_t` and `const nullopt_t &` all compare equal.
std::string normalizeType(llvm::StringRef S) {
  std::string Out;
  while (!S.empty()) {
    llvm::StringRef Word =
        S.take_while([](char C) { return llvm::isAlnum(C) || C == '_'; });
    S = S.drop_front(Word.size());
    // Drop qualifiers (`std::`, a leading `::`) and cv-qualifiers.
    if (S.consume_front("::") || Word == "const" || Word == "volatile")
      continue;
    if (!Word.empty()) {
      Out += Word;
      continue;
    }
    char C = S.front(); // punctuation
    S = S.drop_front();
    if (C != '&' && C != '.' && !llvm::isSpace(C))
      Out += C;
  }
  return Out;
}

AST_MATCHER_P(ParmVarDecl, hasLooselySpelledType, std::string, Spelling) {
  QualType T = Node.getType();
  // Anything dependent (T&&, _Up&&, Args&&...) is "some template parameter":
  // we cannot compare the spelling meaningfully, so accept.
  if (T->isDependentType())
    return true;
  return normalizeType(T.getAsString(Node.getASTContext().getPrintingPolicy())) ==
         Spelling;
}

AST_MATCHER(CXXMethodDecl, isConstructorMethod) {
  return isa<CXXConstructorDecl>(Node);
}

using MethodMatcher = ast_matchers::internal::Matcher<CXXMethodDecl>;

/// Last component of a qualified name: `std::optional` -> `optional`.
llvm::StringRef simpleName(llvm::StringRef Qualified) {
  llvm::StringRef Last = Qualified.rsplit("::").second;
  return Last.empty() ? Qualified : Last;
}

/// The matcher for one attribute string; every part must match.
/// `value` -> named method; `<ClassName>(...)` -> constructor of the target
/// class; `operator=` -> overloaded operator; `operator bool` -> conversion.
MethodMatcher buildMethodMatcher(const MethodSpec &Spec,
                                 llvm::StringRef TargetSimpleName) {
  llvm::StringRef Name = Spec.Name;
  MethodMatcher Result = hasName(Spec.Name);
  if (Name == TargetSimpleName)
    Result = isConstructorMethod();
  else if (Name.consume_front("operator") && !Name.empty() &&
           !llvm::isAlpha(Name.front()) && Name.front() != '_')
    Result = hasOverloadedOperatorName(Name.trim());

  // A trailing pack matches nothing positional beyond its position.
  for (auto [I, P] : llvm::enumerate(Spec.Params)) {
    if (llvm::StringRef(P).ends_with("..."))
      break;
    Result = allOf(Result, hasParameter(I, hasLooselySpelledType(normalizeType(P))));
  }
  return Result;
}

//===----------------------------------------------------------------------===//
// Single-pass visitor over the header AST
//===----------------------------------------------------------------------===//

struct Entry {
  bool IsClass;
  std::string Target;                           ///< qualified target class
  std::optional<MethodMatcher> Method; ///< methods; nullopt: malformed
  bool Valid = true; ///< class entries: non-empty name
  bool Matched = false;
};

class HeaderVisitor : public RecursiveASTVisitor<HeaderVisitor> {
public:
  HeaderVisitor(ASTContext &Ctx, std::vector<Entry> &Entries)
      : Ctx(Ctx), Entries(Entries) {
    for (const Entry &E : Entries) {
      if (E.IsClass ? !E.Valid : !E.Method)
        continue; // malformed: can never match
      ++Pending;
      if (!Targets.count(E.Target))
        Targets.try_emplace(E.Target, hasName("::" + E.Target));
    }
  }

  bool VisitCXXRecordDecl(CXXRecordDecl *RD) {
    if (!RD->isThisDeclarationADefinition() || Pending == 0)
      return Pending != 0;
    for (const auto &[Target, Matcher] : Targets) {
      if (match(cxxRecordDecl(Matcher), *RD, Ctx).empty())
        continue;
      // The target class exists: every class entry naming it is satisfied.
      for (Entry &E : Entries)
        if (E.IsClass && E.Valid && !E.Matched && E.Target == Target)
          markMatched(E);
      // The members of the target are not only the ones it declares itself:
      // implementations commonly inherit them from (template) bases.
      llvm::SmallPtrSet<const CXXRecordDecl *, 8> Seen;
      visitMembers(RD, Target, Seen);
    }
    return Pending != 0;
  }

private:
  void markMatched(Entry &E) {
    E.Matched = true;
    --Pending;
  }

  void visitMembers(const CXXRecordDecl *RD, const std::string &Target,
                    llvm::SmallPtrSetImpl<const CXXRecordDecl *> &Seen) {
    if (!RD || !Seen.insert(RD->getCanonicalDecl()).second || Pending == 0)
      return;
    for (const Decl *D : RD->decls()) {
      if (const auto *FT = dyn_cast<FunctionTemplateDecl>(D))
        D = FT->getTemplatedDecl();
      if (const auto *MD = dyn_cast<CXXMethodDecl>(D))
        runMethod(*MD, Target);
    }
    for (const CXXBaseSpecifier &B : RD->bases()) {
      const CXXRecordDecl *Base = B.getType()->getAsCXXRecordDecl();
      const ClassTemplateDecl *CTD = nullptr;
      if (!Base)
        if (const auto *TST = B.getType()->getAs<TemplateSpecializationType>())
          CTD = dyn_cast_or_null<ClassTemplateDecl>(
              TST->getTemplateName().getAsTemplateDecl());
      if (CTD) {
        Base = CTD->getTemplatedDecl();
        // The primary template may be only forward-declared with all members
        // in partial specializations (as libc++ does for its optional bases).
        // A member of any of them is "a member" for our fuzzy purposes.
        llvm::SmallVector<ClassTemplatePartialSpecializationDecl *, 4> Partials;
        const_cast<ClassTemplateDecl *>(CTD)->getPartialSpecializations(
            Partials);
        for (const ClassTemplatePartialSpecializationDecl *PS : Partials)
          visitMembers(PS->getDefinition(), Target, Seen);
      }
      if (Base && (Base = Base->getDefinition()))
        visitMembers(Base, Target, Seen);
    }
  }

  void runMethod(const CXXMethodDecl &MD, const std::string &Target) {
    for (Entry &E : Entries) {
      if (E.Matched || E.IsClass || !E.Method || E.Target != Target)
        continue;
      if (!match(cxxMethodDecl(*E.Method), MD, Ctx).empty())
        markMatched(E);
    }
  }

  ASTContext &Ctx;
  std::vector<Entry> &Entries;
  std::map<std::string, ast_matchers::internal::Matcher<CXXRecordDecl>> Targets;
  unsigned Pending = 0;
};

std::unique_ptr<ASTUnit> buildHeaderAST(const HeaderASTConfig &Cfg) {
  if (!Cfg.isConfigured())
    return nullptr;

  std::string Code = Cfg.Header.front() == '<'
                         ? "#include " + Cfg.Header + "\n"
                         : "#include \"" + Cfg.Header + "\"\n";
  std::vector<std::string> Args = {
      "-std=" + Cfg.Std, "-fsyntax-only", "-x", "c++", "-isysroot", Cfg.Sysroot,
  };
  // Fail open and silently: problems in the header AST are not the user's.
  IgnoringDiagConsumer Ignore;
  // The driver locates its resource dir (builtin headers) relative to the tool
  // path, so it must be the real executable, not a bare name that would be
  // resolved against the current directory.
  static int Anchor;
  std::string ToolPath =
      llvm::sys::fs::getMainExecutable("clang-tidy", &Anchor);
  std::unique_ptr<ASTUnit> AST = tooling::buildASTFromCodeWithArgs(
      Code, Args, "clang_tidy_analyze_as_header.cc", ToolPath,
      std::make_shared<PCHContainerOperations>(),
      tooling::getClangStripDependencyFileAdjuster(), tooling::FileContentMappings(),
      &Ignore);
  if (!AST || AST->getDiagnostics().hasUncompilableErrorOccurred())
    return nullptr;
  return AST;
}

} // namespace

/// Checks every query against its target class in a header AST built from
/// `Config` (cached per distinct config). Returns one bool per query (true =
/// matched something), or std::nullopt if the config is incomplete or the
/// header AST could not be built.
static std::optional<std::vector<bool>>
validateAgainstHeaderAST(const HeaderASTConfig &Config,
                         llvm::ArrayRef<AttrQuery> Queries) {
  if (Queries.empty())
    return std::vector<bool>();
  if (!Config.isConfigured())
    return std::nullopt; // no (complete) pragma: nothing to validate against

  // Translate each distinct attribute to a matcher exactly once.
  std::vector<Entry> Entries;
  std::map<std::tuple<bool, std::string, std::string>, size_t> Index;
  std::vector<size_t> QueryToEntry;
  for (const AttrQuery &Q : Queries) {
    auto [It, Inserted] =
        Index.try_emplace({Q.IsClass, Q.Target, Q.Text}, Entries.size());
    if (Inserted) {
      Entry E;
      E.IsClass = Q.IsClass;
      E.Target = Q.Target;
      if (Q.IsClass) {
        E.Valid = !Q.Target.empty() && llvm::StringRef(Q.Text).trim() == Q.Target;
      } else if (std::optional<MethodSpec> Spec = parseMethodSpec(Q.Text)) {
        E.Method = buildMethodMatcher(*Spec, simpleName(Q.Target));
      }
      Entries.push_back(std::move(E));
    }
    QueryToEntry.push_back(It->second);
  }

  {
    static std::map<HeaderASTConfig, std::unique_ptr<ASTUnit>> Cache;
    auto It = Cache.find(Config);
    if (It == Cache.end())
      It = Cache.emplace(Config, buildHeaderAST(Config)).first;
    ASTUnit *AST = It->second.get();
    if (!AST)
      return std::nullopt;

    HeaderVisitor V(AST->getASTContext(), Entries);
    V.TraverseDecl(AST->getASTContext().getTranslationUnitDecl());
  }

  std::vector<bool> Result;
  Result.reserve(Queries.size());
  for (size_t EntryIdx : QueryToEntry)
    Result.push_back(Entries[EntryIdx].Matched);
  return Result;
}

namespace {

/// `#pragma clang_tidy_analyze_as_header sysroot("<dir>") header("<optional>") std("c++23")`
/// All parameters are mandatory and must appear in this order; nothing is
/// defaulted or inferred. They tell the attribute validator how to build its
/// header AST. Anything else is reported by the check.
class AnalyzeAsHeaderPragmaHandler : public PragmaHandler {
public:
  AnalyzeAsHeaderPragmaHandler(std::shared_ptr<HeaderASTConfig> Config)
      : PragmaHandler("clang_tidy_analyze_as_header"), Config(std::move(Config)) {}

  void HandlePragma(Preprocessor &PP, PragmaIntroducer,
                    Token &FirstToken) override {
    HeaderASTConfig NewConfig;
    Token Tok;
    if (lexParam(PP, Tok, "sysroot", NewConfig.Sysroot) &&
        lexParam(PP, Tok, "header", NewConfig.Header) &&
        lexParam(PP, Tok, "std", NewConfig.Std)) {
      PP.LexUnexpandedToken(Tok);
      if (Tok.is(tok::eod)) {
        *Config = std::move(NewConfig);
        return;
      }
    }
    // Reported by the check (a tidy diagnostic) once matching starts.
    Config->BadPragmaLocs.push_back(FirstToken.getLocation());
  }

private:
  /// Lexes `Name("<string>")` into `Value`.
  static bool lexParam(Preprocessor &PP, Token &Tok, StringRef Name,
                       std::string &Value) {
    PP.LexUnexpandedToken(Tok);
    if (!Tok.is(tok::identifier) || Tok.getIdentifierInfo()->getName() != Name)
      return false;
    PP.LexUnexpandedToken(Tok);
    if (Tok.isNot(tok::l_paren))
      return false;
    PP.LexUnexpandedToken(Tok);
    if (!tok::isStringLiteral(Tok.getKind()))
      return false;
    Value = StringLiteralParser(Tok, PP).GetString().str();
    PP.LexUnexpandedToken(Tok);
    return Tok.is(tok::r_paren);
  }

  std::shared_ptr<HeaderASTConfig> Config;
};

/// Collects the `analyze_as_*` attributes written in user code, in one walk.
class AnalyzeAsAttrCollector : public RecursiveASTVisitor<AnalyzeAsAttrCollector> {
public:
  struct Found {
    AttrQuery Query;
    SourceLocation Loc;
    llvm::StringRef Spelling;
  };

  explicit AnalyzeAsAttrCollector(const SourceManager &SM) : SM(SM) {}

  bool TraverseDecl(Decl *D) {
    // Skip system headers wholesale: only the user's own annotations are
    // validated (and the header's std::optional lives in one).
    if (D && !isa<TranslationUnitDecl>(D) &&
        SM.isInSystemHeader(D->getLocation()))
      return true;
    return RecursiveASTVisitor::TraverseDecl(D);
  }

  bool VisitDecl(Decl *D) {
    for (const auto *A : D->specific_attrs<AnalyzeAsClassAttr>()) {
      std::string Name = A->getClassName().str();
      Results.push_back(
          {{/*IsClass=*/true, Name, Name}, A->getLocation(), "analyze_as_class"});
    }
    for (const auto *A : D->specific_attrs<AnalyzeAsMethodAttr>()) {
      // A method refers to the class its enclosing record is annotated as.
      const auto *MD = dyn_cast<CXXMethodDecl>(D);
      const auto *Class =
          MD ? MD->getParent()->getAttr<AnalyzeAsClassAttr>() : nullptr;
      if (!Class)
        continue; // no target class to validate against
      Results.push_back({{/*IsClass=*/false, A->getMethodName().str(),
                          Class->getClassName().str()},
                         A->getLocation(), "analyze_as_method"});
    }
    return true;
  }

  std::vector<Found> Results;

private:
  const SourceManager &SM;
};

/// Validates the `analyze_as_*` attributes of the TU once it is complete.
class AnalyzeAsAttrValidationConsumer : public ASTConsumer {
public:
  AnalyzeAsAttrValidationConsumer(ClangTidyContext &Context,
                                  std::shared_ptr<HeaderASTConfig> Config)
      : Context(Context), Config(std::move(Config)) {}

  void HandleTranslationUnit(ASTContext &Ctx) override {
    if (Ctx.getDiagnostics().hasUncompilableErrorOccurred())
      return;
    for (SourceLocation Loc : Config->BadPragmaLocs)
      report(Loc,
             "malformed 'clang_tidy_analyze_as_header'; expected "
             "sysroot(\"...\") header(\"...\") std(\"...\"); pragma ignored");

    AnalyzeAsAttrCollector Collector(Ctx.getSourceManager());
    Collector.TraverseDecl(Ctx.getTranslationUnitDecl());
    if (Collector.Results.empty())
      return; // Nothing annotated: never pay for the header AST.

    llvm::SmallVector<AttrQuery> Queries;
    for (const auto &F : Collector.Results)
      Queries.push_back(F.Query);

    // Fails open: no header AST means no diagnostics.
    std::optional<std::vector<bool>> Matched =
        validateAgainstHeaderAST(*Config, Queries);
    if (!Matched)
      return;

    for (size_t I = 0; I != Matched->size(); ++I)
      if (!(*Matched)[I])
        report(Collector.Results[I].Loc,
               llvm::formatv("'{0}' argument \"{1}\" does not match anything "
                             "in {2}; the attribute is likely malformed",
                             Collector.Results[I].Spelling, Queries[I].Text,
                             Queries[I].Target)
                   .str());
  }

private:
  void report(SourceLocation Loc, const std::string &Message) {
    Context.diag("clang-tidy-config", Loc, "%0") << Message;
  }

  ClangTidyContext &Context;
  std::shared_ptr<HeaderASTConfig> Config;
};

} // namespace

std::unique_ptr<ASTConsumer>
createAnalyzeAsAttrValidationConsumer(ClangTidyContext &Context,
                                      Preprocessor &PP) {
  auto Config = std::make_shared<HeaderASTConfig>();
  PP.AddPragmaHandler(new AnalyzeAsHeaderPragmaHandler(Config));
  return std::make_unique<AnalyzeAsAttrValidationConsumer>(Context,
                                                           std::move(Config));
}

} // namespace clang::tidy
