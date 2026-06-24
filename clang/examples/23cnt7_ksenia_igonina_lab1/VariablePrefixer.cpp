#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace {

enum class VarCategory {
  Global,
  StaticLocal,
  Local,
  Param
};

// Храним отображение "старое имя -> новое имя с префиксом"
std::map<std::string, std::string> RenameMap;

class VariablePrefixerHandler : public MatchFinder::MatchCallback {
public:
  VariablePrefixerHandler(Rewriter &Rewrite) : Rewrite(Rewrite) {}

  void run(const MatchFinder::MatchResult &Result) override {
    // Обработка объявлений (добавляем в RenameMap и заменяем)
    if (const auto *PVD = Result.Nodes.getNodeAs<ParmVarDecl>("param")) {
      handleDecl(PVD, VarCategory::Param, Result);
    }
    else if (const auto *VD = Result.Nodes.getNodeAs<VarDecl>("global")) {
      handleDecl(VD, VarCategory::Global, Result);
    }
    else if (const auto *VD = Result.Nodes.getNodeAs<VarDecl>("staticLocal")) {
      handleDecl(VD, VarCategory::StaticLocal, Result);
    }
    else if (const auto *VD = Result.Nodes.getNodeAs<VarDecl>("local")) {
      handleDecl(VD, VarCategory::Local, Result);
    }
    // Обработка использований (берём префикс из RenameMap)
    else if (const auto *DRE = Result.Nodes.getNodeAs<DeclRefExpr>("use")) {
      handleUse(DRE, Result);
    }
  }

private:
  void handleDecl(const NamedDecl *D, VarCategory Cat,
                  const MatchFinder::MatchResult &Result) {
    std::string OrigName = D->getNameAsString();
    if (OrigName.empty()) return;

    // Если уже есть в мапе, не дублируем
    if (RenameMap.find(OrigName) != RenameMap.end())
      return;

    std::string Prefix;
    switch (Cat) {
      case VarCategory::Global:   Prefix = "global_"; break;
      case VarCategory::StaticLocal: Prefix = "static_"; break;
      case VarCategory::Local:    Prefix = "local_"; break;
      case VarCategory::Param:    Prefix = "param_"; break;
    }

    std::string NewName = Prefix + OrigName;
    RenameMap[OrigName] = NewName;

    // Заменяем в объявлении
    SourceLocation Loc = D->getLocation();
    if (Loc.isInvalid()) return;

    Loc = Result.SourceManager->getSpellingLoc(Loc);
    if (!Result.SourceManager->isInMainFile(Loc))
      return;

    unsigned NameLen = Lexer::MeasureTokenLength(Loc,
                                                  *Result.SourceManager,
                                                  Result.Context->getLangOpts());
    if (NameLen != OrigName.length())
      NameLen = OrigName.length();

    Rewrite.ReplaceText(Loc, NameLen, NewName);
  }

  void handleUse(const DeclRefExpr *DRE,
                 const MatchFinder::MatchResult &Result) {
    const ValueDecl *D = DRE->getDecl();
    if (!D) return;
    std::string OrigName = D->getNameAsString();
    if (OrigName.empty()) return;

    auto It = RenameMap.find(OrigName);
    if (It == RenameMap.end()) return; // не нашли объявление

    const std::string &NewName = It->second;

    SourceLocation Loc = DRE->getLocation();
    if (Loc.isInvalid()) return;

    Loc = Result.SourceManager->getSpellingLoc(Loc);
    if (!Result.SourceManager->isInMainFile(Loc))
      return;

    unsigned NameLen = Lexer::MeasureTokenLength(Loc,
                                                  *Result.SourceManager,
                                                  Result.Context->getLangOpts());
    if (NameLen != OrigName.length())
      NameLen = OrigName.length();

    Rewrite.ReplaceText(Loc, NameLen, NewName);
  }

  Rewriter &Rewrite;
};

class VariablePrefixerAction : public PluginASTAction {
private:
  Rewriter Rewrite;
  MatchFinder Finder;
  VariablePrefixerHandler Handler{Rewrite};

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                  StringRef InFile) override {
    Rewrite.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    RenameMap.clear(); // очищаем перед каждым файлом

    // Матчеры для объявлений
    Finder.addMatcher(parmVarDecl().bind("param"), &Handler);
    Finder.addMatcher(varDecl(hasGlobalStorage(),
                              unless(hasAncestor(functionDecl()))).bind("global"),
                      &Handler);
    Finder.addMatcher(varDecl(hasStaticStorageDuration(),
                              hasAncestor(functionDecl())).bind("staticLocal"),
                      &Handler);
    Finder.addMatcher(varDecl(hasAutomaticStorageDuration(),
                              unless(parmVarDecl()),
                              hasAncestor(functionDecl())).bind("local"),
                      &Handler);

    // Матчеры для использований (DeclRefExpr)
    // Для глобальных переменных
    Finder.addMatcher(declRefExpr(to(varDecl(hasGlobalStorage(),
                                              unless(hasAncestor(functionDecl()))))).bind("use"),
                      &Handler);
    // Для статических локальных
    Finder.addMatcher(declRefExpr(to(varDecl(hasStaticStorageDuration(),
                                              hasAncestor(functionDecl())))).bind("use"),
                      &Handler);
    // Для обычных локальных (не параметры)
    Finder.addMatcher(declRefExpr(to(varDecl(hasAutomaticStorageDuration(),
                                              unless(parmVarDecl()),
                                              hasAncestor(functionDecl())))).bind("use"),
                      &Handler);
    // Для параметров
    Finder.addMatcher(declRefExpr(to(parmVarDecl())).bind("use"), &Handler);

    return Finder.newASTConsumer();
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    return true;
  }

  void EndSourceFileAction() override {
    SourceManager &SM = Rewrite.getSourceMgr();
    FileID MainFileID = SM.getMainFileID();
    Rewrite.getEditBuffer(MainFileID).write(llvm::outs());
  }
};

} // namespace

static FrontendPluginRegistry::Add<VariablePrefixerAction>
X("variable-prefixer", "Adds prefixes to variable names based on their category");
