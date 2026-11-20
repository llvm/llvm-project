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
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

using namespace clang::ast_matchers;

namespace clang::tidy {

/// How to build the header AST (the throw-away AST of the standard library
/// header that the annotated types are supposed to mimic).
///
/// Nothing here is configured by the user: it is a copy of the compiler
/// invocation of the TU being analysed, so the header is found exactly where
/// the project's own build finds it (sysroot, -isystem, -stdlib, target, ...).
struct HeaderASTConfig {
  /// Copy of the analysed TU's invocation, taken before parsing starts.
  std::shared_ptr<CompilerInvocation> Invocation;
  /// Working directory of the analysed TU (relative include paths in a
  /// compilation database are relative to it).
  std::string WorkingDir;
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

/// Every supported `analyze_as_class` argument. To support another class, add
/// a row here once a check models it.
constexpr AnalyzeAsTarget KnownTargets[] = {
    {"std::optional", "<optional>", "bugprone-unchecked-optional-access"},
    // Example (not supported yet):
    // {"std::vector", "<vector>", "<check-that-models-std-vector>"},
};

const AnalyzeAsTarget *findTarget(llvm::StringRef Name,
                                  llvm::ArrayRef<AnalyzeAsTarget> Table =
                                      KnownTargets) {
  for (const AnalyzeAsTarget &T : Table)
    if (T.Name == Name)
      return &T;
  return nullptr;
}

std::optional<llvm::StringRef> headerForTarget(llvm::StringRef Target) {
  if (const AnalyzeAsTarget *T = findTarget(Target))
    return T->Header;
  return std::nullopt;
}

/// Derives the invocation for the header AST from the analysed TU's one:
/// same header search, target and macros; the main file replaced by `Path`;
/// anything that would write files or pull in project code dropped.
std::shared_ptr<CompilerInvocation>
makeHeaderInvocation(const CompilerInvocation &TU, llvm::StringRef Path) {
  auto CI = std::make_shared<CompilerInvocation>(TU);

  FrontendOptions &FE = CI->getFrontendOpts();
  FE.Inputs = {FrontendInputFile(Path, Language::CXX)};
  FE.ProgramAction = frontend::ParseSyntaxOnly;
  FE.OutputFile.clear();
  CI->getDependencyOutputOpts() = DependencyOutputOptions();
  CI->getCodeGenOpts().MainFileName.clear();

  PreprocessorOptions &PP = CI->getPreprocessorOpts();
  PP.Includes.clear(); // -include of project headers
  PP.ImplicitPCHInclude.clear();

  // The target class may not exist below its standard (std::optional is
  // C++17), so raise the standard of the header AST only.
  LangOptions &LO = CI->getLangOpts();
  if (!LO.CPlusPlus17) {
    std::vector<std::string> Ignored;
    LangOptions::setLangDefaults(LO, Language::CXX,
                                 llvm::Triple(CI->getTargetOpts().Triple),
                                 Ignored, LangStandard::lang_cxx17);
  }
  return CI;
}

std::unique_ptr<ASTUnit> buildHeaderAST(std::shared_ptr<CompilerInvocation> CI,
                                        llvm::StringRef Path,
                                        llvm::StringRef Code,
                                        llvm::StringRef WorkingDir) {
  // The main file only exists in memory; everything else comes from disk.
  auto Overlay = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(
      llvm::vfs::getRealFileSystem());
  auto InMemory = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  Overlay->pushOverlay(InMemory);
  if (!WorkingDir.empty())
    Overlay->setCurrentWorkingDirectory(WorkingDir);
  InMemory->addFile(Path, 0, llvm::MemoryBuffer::getMemBufferCopy(Code));

  auto Files = llvm::makeIntrusiveRefCnt<FileManager>(CI->getFileSystemOpts(),
                                                      Overlay);
  // Fail open and silently: problems in the header AST are not the user's.
  auto DiagOpts = std::make_shared<DiagnosticOptions>();
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(*Overlay, *DiagOpts,
                                          new IgnoringDiagConsumer(),
                                          /*ShouldOwnClient=*/true);
  std::unique_ptr<ASTUnit> AST = ASTUnit::LoadFromCompilerInvocation(
      std::move(CI), std::make_shared<PCHContainerOperations>(), DiagOpts,
      Diags, Files);
  if (!AST || AST->getDiagnostics().hasUncompilableErrorOccurred())
    return nullptr;
  return AST;
}

} // namespace

/// Checks every query against its target class in a header AST built with the
/// analysed TU's own compiler settings (cached per distinct setting). Returns
/// one bool per query (true = matched something), or std::nullopt if the
/// header AST could not be built.
static std::optional<std::vector<bool>>
validateAgainstHeaderAST(const HeaderASTConfig &Config,
                         llvm::ArrayRef<AttrQuery> Queries) {
  if (Queries.empty())
    return std::vector<bool>();
  if (!Config.Invocation || !Config.Invocation->getLangOpts().CPlusPlus)
    return std::nullopt;

  // One #include per distinct header the queries need.
  std::string Code;
  llvm::StringSet<> Seen;
  for (const AttrQuery &Q : Queries)
    if (std::optional<llvm::StringRef> H = headerForTarget(Q.Target))
      if (Seen.insert(*H).second)
        Code += ("#include " + *H + "\n").str();
  if (Code.empty())
    return std::nullopt; // no known target: nothing to validate against

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
    llvm::SmallString<256> Path(Config.WorkingDir);
    llvm::sys::path::append(Path, "__clang_tidy_analyze_as_header.cpp");
    std::shared_ptr<CompilerInvocation> CI =
        makeHeaderInvocation(*Config.Invocation, Path);

    // Key: everything that affects how the header is found and parsed.
    std::string Key = Config.WorkingDir + '\n' + Code;
    for (const std::string &Arg : CI->getCC1CommandLine())
      Key += '\n' + Arg;

    // clang-tidy may run TUs on several threads (e.g. inside clangd), and the
    // cached ASTs are shared, so both the cache and the traversal are locked.
    static std::mutex CacheMutex;
    static std::map<std::string, std::unique_ptr<ASTUnit>> Cache;
    std::lock_guard<std::mutex> Lock(CacheMutex);
    auto It = Cache.find(Key);
    if (It == Cache.end())
      It = Cache
               .emplace(Key, buildHeaderAST(std::move(CI), Path, Code,
                                            Config.WorkingDir))
               .first;
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
                                  HeaderASTConfig Config)
      : Context(Context), Config(std::move(Config)) {}

  void HandleTranslationUnit(ASTContext &Ctx) override {
    if (Ctx.getDiagnostics().hasUncompilableErrorOccurred())
      return;
    AnalyzeAsAttrCollector Collector(Ctx.getSourceManager());
    Collector.TraverseDecl(Ctx.getTranslationUnitDecl());
    if (Collector.Results.empty())
      return; // Nothing annotated: never pay for the header AST.

    // Decide what to do with each attribute based on its target class:
    // - unknown class: report the class attribute (under an enabled check that
    //   uses analyze_as, if any) and stay silent on its methods;
    // - known class whose check is disabled: nothing uses it, stay silent;
    // - known class whose check is enabled: validate against the header AST.
    auto IsEnabled = [this](llvm::StringRef Check) {
      return Context.isCheckEnabled(Check);
    };
    std::vector<const AnalyzeAsAttrCollector::Found *> ToValidate;
    for (const auto &F : Collector.Results) {
      AnalyzeAsDecision D =
          classifyAnalyzeAsTarget(F.Query.Target, KnownTargets, IsEnabled);
      switch (D.Action) {
      case AnalyzeAsDecision::Validate:
        ToValidate.push_back(&F);
        break;
      case AnalyzeAsDecision::ReportUnknown:
        if (F.Query.IsClass)
          reportUnknownClass(F, D.Check);
        break;
      case AnalyzeAsDecision::Silent:
        break;
      }
    }
    if (ToValidate.empty())
      return;

    llvm::SmallVector<AttrQuery> Queries;
    for (const auto *F : ToValidate)
      Queries.push_back(F->Query);

    // Fails open: no header AST means no diagnostics.
    std::optional<std::vector<bool>> Matched =
        validateAgainstHeaderAST(Config, Queries);
    if (!Matched)
      return;

    for (size_t I = 0; I != Matched->size(); ++I)
      if (!(*Matched)[I])
        report("clang-tidy-config", ToValidate[I]->Loc,
               llvm::formatv("'{0}' argument \"{1}\" does not match anything "
                             "in {2}; the attribute is likely malformed",
                             ToValidate[I]->Spelling, Queries[I].Text,
                             Queries[I].Target)
                   .str());
  }

private:
  /// An `analyze_as_class` naming a class no check supports, reported under
  /// `Check` so it shows up (and can be suppressed) like that check's own
  /// warnings.
  void reportUnknownClass(const AnalyzeAsAttrCollector::Found &F,
                          llvm::StringRef Check) {
    std::string Supported;
    for (const AnalyzeAsTarget &T : KnownTargets)
      Supported += (Supported.empty() ? "" : ", ") + T.Name.str();
    report(Check, F.Loc,
           llvm::formatv("'analyze_as_class' argument \"{0}\" is not a "
                         "supported class; supported: {1}",
                         F.Query.Text, Supported)
               .str());
  }

  void report(llvm::StringRef CheckName, SourceLocation Loc,
              const std::string &Message) {
    Context.diag(CheckName, Loc, "%0") << Message;
  }

  ClangTidyContext &Context;
  HeaderASTConfig Config;
};

} // namespace

AnalyzeAsDecision
classifyAnalyzeAsTarget(llvm::StringRef Target,
                        llvm::ArrayRef<AnalyzeAsTarget> Table,
                        llvm::function_ref<bool(llvm::StringRef)> IsEnabled) {
  if (const AnalyzeAsTarget *T = findTarget(Target, Table)) {
    // Supported class: only worth validating if its check will use it.
    if (IsEnabled(T->Check))
      return {AnalyzeAsDecision::Validate, T->Check};
    return {AnalyzeAsDecision::Silent, {}};
  }
  // Unsupported class: report it under the first enabled check that uses
  // analyze_as attributes; if there is none, it is irrelevant to this run.
  for (const AnalyzeAsTarget &T : Table)
    if (IsEnabled(T.Check))
      return {AnalyzeAsDecision::ReportUnknown, T.Check};
  return {AnalyzeAsDecision::Silent, {}};
}

std::unique_ptr<ASTConsumer>
createAnalyzeAsAttrValidationConsumer(ClangTidyContext &Context,
                                      CompilerInstance &Compiler) {
  // Snapshot the TU's settings now; the header AST is built (lazily, only if
  // the TU has annotations) from a copy of them.
  HeaderASTConfig Config;
  Config.Invocation =
      std::make_shared<CompilerInvocation>(Compiler.getInvocation());
  if (llvm::ErrorOr<std::string> CWD =
          Compiler.getVirtualFileSystem().getCurrentWorkingDirectory())
    Config.WorkingDir = *CWD;
  return std::make_unique<AnalyzeAsAttrValidationConsumer>(Context,
                                                           std::move(Config));
}

} // namespace clang::tidy
