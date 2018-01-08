//===--- MakeSmartPtrCheck.cpp - clang-tidy--------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MakeSharedCheck.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

namespace {

constexpr char StdMemoryHeader[] = "memory";

std::string GetNewExprName(const CXXNewExpr *NewExpr,
                           const SourceManager &SM,
                           const LangOptions &Lang) {
  StringRef WrittenName = Lexer::getSourceText(
      CharSourceRange::getTokenRange(
          NewExpr->getAllocatedTypeSourceInfo()->getTypeLoc().getSourceRange()),
      SM, Lang);
  if (NewExpr->isArray()) {
    return WrittenName.str() + "[]";
  }
  return WrittenName.str();
}

} // namespace

const char MakeSmartPtrCheck::PointerType[] = "pointerType";
const char MakeSmartPtrCheck::ConstructorCall[] = "constructorCall";
const char MakeSmartPtrCheck::ResetCall[] = "resetCall";
const char MakeSmartPtrCheck::NewExpression[] = "newExpression";

MakeSmartPtrCheck::MakeSmartPtrCheck(StringRef Name,
                                     ClangTidyContext* Context,
                                     StringRef MakeSmartPtrFunctionName)
    : ClangTidyCheck(Name, Context),
      IncludeStyle(utils::IncludeSorter::parseIncludeStyle(
          Options.getLocalOrGlobal("IncludeStyle", "llvm"))),
      MakeSmartPtrFunctionHeader(
          Options.get("MakeSmartPtrFunctionHeader", StdMemoryHeader)),
      MakeSmartPtrFunctionName(
          Options.get("MakeSmartPtrFunction", MakeSmartPtrFunctionName)),
      IgnoreMacros(Options.getLocalOrGlobal("IgnoreMacros", true)) {}

void MakeSmartPtrCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", IncludeStyle);
  Options.store(Opts, "MakeSmartPtrFunctionHeader", MakeSmartPtrFunctionHeader);
  Options.store(Opts, "MakeSmartPtrFunction", MakeSmartPtrFunctionName);
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void MakeSmartPtrCheck::registerPPCallbacks(CompilerInstance &Compiler) {
  if (getLangOpts().CPlusPlus11) {
    Inserter.reset(new utils::IncludeInserter(
        Compiler.getSourceManager(), Compiler.getLangOpts(), IncludeStyle));
    Compiler.getPreprocessor().addPPCallbacks(Inserter->CreatePPCallbacks());
  }
}

void MakeSmartPtrCheck::registerMatchers(ast_matchers::MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus11)
    return;

  // Calling make_smart_ptr from within a member function of a type with a
  // private or protected constructor would be ill-formed.
  auto CanCallCtor = unless(has(ignoringImpCasts(
      cxxConstructExpr(hasDeclaration(decl(unless(isPublic())))))));

  Finder->addMatcher(
      cxxBindTemporaryExpr(has(ignoringParenImpCasts(
          cxxConstructExpr(
              hasType(getSmartPointerTypeMatcher()), argumentCountIs(1),
              hasArgument(0,
                          cxxNewExpr(hasType(pointsTo(qualType(hasCanonicalType(
                                         equalsBoundNode(PointerType))))),
                                     CanCallCtor)
                              .bind(NewExpression)),
              unless(isInTemplateInstantiation()))
              .bind(ConstructorCall)))),
      this);

  Finder->addMatcher(
      cxxMemberCallExpr(
          thisPointerType(getSmartPointerTypeMatcher()),
          callee(cxxMethodDecl(hasName("reset"))),
          hasArgument(0, cxxNewExpr(CanCallCtor).bind(NewExpression)),
          unless(isInTemplateInstantiation()))
          .bind(ResetCall),
      this);
}

void MakeSmartPtrCheck::check(const MatchFinder::MatchResult &Result) {
  // 'smart_ptr' refers to 'std::shared_ptr' or 'std::unique_ptr' or other
  // pointer, 'make_smart_ptr' refers to 'std::make_shared' or
  // 'std::make_unique' or other function that creates smart_ptr.

  SourceManager &SM = *Result.SourceManager;
  const auto *Construct =
      Result.Nodes.getNodeAs<CXXConstructExpr>(ConstructorCall);
  const auto *Reset = Result.Nodes.getNodeAs<CXXMemberCallExpr>(ResetCall);
  const auto *Type = Result.Nodes.getNodeAs<QualType>(PointerType);
  const auto *New = Result.Nodes.getNodeAs<CXXNewExpr>(NewExpression);

  if (New->getNumPlacementArgs() != 0)
    return;

  if (Construct)
    checkConstruct(SM, Construct, Type, New);
  else if (Reset)
    checkReset(SM, Reset, New);
}

void MakeSmartPtrCheck::checkConstruct(SourceManager &SM,
                                       const CXXConstructExpr *Construct,
                                       const QualType *Type,
                                       const CXXNewExpr *New) {
  SourceLocation ConstructCallStart = Construct->getExprLoc();
  bool InMacro = ConstructCallStart.isMacroID();

  if (InMacro && IgnoreMacros) {
    return;
  }

  bool Invalid = false;
  StringRef ExprStr = Lexer::getSourceText(
      CharSourceRange::getCharRange(
          ConstructCallStart, Construct->getParenOrBraceRange().getBegin()),
      SM, getLangOpts(), &Invalid);
  if (Invalid)
    return;

  auto Diag = diag(ConstructCallStart, "use %0 instead")
              << MakeSmartPtrFunctionName;

  // Disable the fix in macros.
  if (InMacro) {
    return;
  }

  if (!replaceNew(Diag, New, SM)) {
    return;
  }

  // Find the location of the template's left angle.
  size_t LAngle = ExprStr.find("<");
  SourceLocation ConstructCallEnd;
  if (LAngle == StringRef::npos) {
    // If the template argument is missing (because it is part of the alias)
    // we have to add it back.
    ConstructCallEnd = ConstructCallStart.getLocWithOffset(ExprStr.size());
    Diag << FixItHint::CreateInsertion(
        ConstructCallEnd,
        "<" + GetNewExprName(New, SM, getLangOpts()) + ">");
  } else {
    ConstructCallEnd = ConstructCallStart.getLocWithOffset(LAngle);
  }

  Diag << FixItHint::CreateReplacement(
      CharSourceRange::getCharRange(ConstructCallStart, ConstructCallEnd),
      MakeSmartPtrFunctionName);

  // If the smart_ptr is built with brace enclosed direct initialization, use
  // parenthesis instead.
  if (Construct->isListInitialization()) {
    SourceRange BraceRange = Construct->getParenOrBraceRange();
    Diag << FixItHint::CreateReplacement(
        CharSourceRange::getCharRange(
            BraceRange.getBegin(), BraceRange.getBegin().getLocWithOffset(1)),
        "(");
    Diag << FixItHint::CreateReplacement(
        CharSourceRange::getCharRange(BraceRange.getEnd(),
                                      BraceRange.getEnd().getLocWithOffset(1)),
        ")");
  }

  insertHeader(Diag, SM.getFileID(ConstructCallStart));
}

void MakeSmartPtrCheck::checkReset(SourceManager &SM,
                                   const CXXMemberCallExpr *Reset,
                                   const CXXNewExpr *New) {
  const auto *Expr = cast<MemberExpr>(Reset->getCallee());
  SourceLocation OperatorLoc = Expr->getOperatorLoc();
  SourceLocation ResetCallStart = Reset->getExprLoc();
  SourceLocation ExprStart = Expr->getLocStart();
  SourceLocation ExprEnd =
      Lexer::getLocForEndOfToken(Expr->getLocEnd(), 0, SM, getLangOpts());

  bool InMacro = ExprStart.isMacroID();

  if (InMacro && IgnoreMacros) {
    return;
  }

  // There are some cases where we don't have operator ("." or "->") of the
  // "reset" expression, e.g. call "reset()" method directly in the subclass of
  // "std::unique_ptr<>". We skip these cases.
  if (OperatorLoc.isInvalid()) {
    return;
  }

  auto Diag = diag(ResetCallStart, "use %0 instead")
              << MakeSmartPtrFunctionName;

  // Disable the fix in macros.
  if (InMacro) {
    return;
  }

  if (!replaceNew(Diag, New, SM)) {
    return;
  }

  Diag << FixItHint::CreateReplacement(
      CharSourceRange::getCharRange(OperatorLoc, ExprEnd),
      (llvm::Twine(" = ") + MakeSmartPtrFunctionName + "<" +
       GetNewExprName(New, SM, getLangOpts()) + ">")
          .str());

  if (Expr->isArrow())
    Diag << FixItHint::CreateInsertion(ExprStart, "*");

  insertHeader(Diag, SM.getFileID(OperatorLoc));
}

bool MakeSmartPtrCheck::replaceNew(DiagnosticBuilder &Diag,
                                   const CXXNewExpr *New,
                                   SourceManager& SM) {
  SourceLocation NewStart = New->getSourceRange().getBegin();
  SourceLocation NewEnd = New->getSourceRange().getEnd();

  std::string ArraySizeExpr;
  if (const auto* ArraySize = New->getArraySize()) {
    ArraySizeExpr = Lexer::getSourceText(CharSourceRange::getTokenRange(
                                             ArraySize->getSourceRange()),
                                         SM, getLangOpts())
                        .str();
  }

  switch (New->getInitializationStyle()) {
  case CXXNewExpr::NoInit: {
    if (ArraySizeExpr.empty()) {
      Diag << FixItHint::CreateRemoval(SourceRange(NewStart, NewEnd));
    } else {
      // New array expression without written initializer:
      //   smart_ptr<Foo[]>(new Foo[5]);
      Diag << FixItHint::CreateReplacement(SourceRange(NewStart, NewEnd),
                                           ArraySizeExpr);
    }
    break;
  }
  case CXXNewExpr::CallInit: {
    // FIXME: Add fixes for constructors with parameters that can be created
    // with a C++11 braced-init-list (e.g. std::vector, std::map).
    // Unlike ordinal cases, braced list can not be deduced in
    // std::make_smart_ptr, we need to specify the type explicitly in the fixes:
    //   struct S { S(std::initializer_list<int>, int); };
    //   struct S2 { S2(std::vector<int>); };
    //   smart_ptr<S>(new S({1, 2, 3}, 1));  // C++98 call-style initialization
    //   smart_ptr<S>(new S({}, 1));
    //   smart_ptr<S2>(new S2({1})); // implicit conversion:
    //                               //   std::initializer_list => std::vector
    // The above samples have to be replaced with:
    //   std::make_smart_ptr<S>(std::initializer_list<int>({1, 2, 3}), 1);
    //   std::make_smart_ptr<S>(std::initializer_list<int>({}), 1);
    //   std::make_smart_ptr<S2>(std::vector<int>({1}));
    if (const auto *CE = New->getConstructExpr()) {
      for (const auto *Arg : CE->arguments()) {
        if (isa<CXXStdInitializerListExpr>(Arg)) {
          return false;
        }
        // Check the implicit conversion from the std::initializer_list type to
        // a class type.
        if (const auto *ImplicitCE = dyn_cast<CXXConstructExpr>(Arg)) {
          if (ImplicitCE->isStdInitListInitialization()) {
            return false;
          }
        }
      }
    }
    if (ArraySizeExpr.empty()) {
      SourceRange InitRange = New->getDirectInitRange();
      Diag << FixItHint::CreateRemoval(
          SourceRange(NewStart, InitRange.getBegin()));
      Diag << FixItHint::CreateRemoval(SourceRange(InitRange.getEnd(), NewEnd));
    }
    else {
      // New array expression with default/value initialization:
      //   smart_ptr<Foo[]>(new int[5]());
      //   smart_ptr<Foo[]>(new Foo[5]());
      Diag << FixItHint::CreateReplacement(SourceRange(NewStart, NewEnd),
                                           ArraySizeExpr);
    }
    break;
  }
  case CXXNewExpr::ListInit: {
    // Range of the substring that we do not want to remove.
    SourceRange InitRange;
    if (const auto *NewConstruct = New->getConstructExpr()) {
      if (NewConstruct->isStdInitListInitialization()) {
        // FIXME: Add fixes for direct initialization with the initializer-list
        // constructor. Similar to the above CallInit case, the type has to be
        // specified explicitly in the fixes.
        //   struct S { S(std::initializer_list<int>); };
        //   smart_ptr<S>(new S{1, 2, 3});  // C++11 direct list-initialization
        //   smart_ptr<S>(new S{});  // use initializer-list consturctor
        // The above cases have to be replaced with:
        //   std::make_smart_ptr<S>(std::initializer_list<int>({1, 2, 3}));
        //   std::make_smart_ptr<S>(std::initializer_list<int>({}));
        return false;
      } else {
        // Direct initialization with ordinary constructors.
        //   struct S { S(int x); S(); };
        //   smart_ptr<S>(new S{5});
        //   smart_ptr<S>(new S{}); // use default constructor
        // The arguments in the initialization list are going to be forwarded to
        // the constructor, so this has to be replaced with:
        //   std::make_smart_ptr<S>(5);
        //   std::make_smart_ptr<S>();
        InitRange = SourceRange(
            NewConstruct->getParenOrBraceRange().getBegin().getLocWithOffset(1),
            NewConstruct->getParenOrBraceRange().getEnd().getLocWithOffset(-1));
      }
    } else {
      // Aggregate initialization.
      //   smart_ptr<Pair>(new Pair{first, second});
      // Has to be replaced with:
      //   smart_ptr<Pair>(Pair{first, second});
      InitRange = SourceRange(
          New->getAllocatedTypeSourceInfo()->getTypeLoc().getLocStart(),
          New->getInitializer()->getSourceRange().getEnd());
    }
    Diag << FixItHint::CreateRemoval(
        CharSourceRange::getCharRange(NewStart, InitRange.getBegin()));
    Diag << FixItHint::CreateRemoval(
        SourceRange(InitRange.getEnd().getLocWithOffset(1), NewEnd));
    break;
  }
  }
  return true;
}

void MakeSmartPtrCheck::insertHeader(DiagnosticBuilder &Diag, FileID FD) {
  if (MakeSmartPtrFunctionHeader.empty()) {
    return;
  }
  if (auto IncludeFixit = Inserter->CreateIncludeInsertion(
          FD, MakeSmartPtrFunctionHeader,
          /*IsAngled=*/MakeSmartPtrFunctionHeader == StdMemoryHeader)) {
    Diag << *IncludeFixit;
  }
}

} // namespace modernize
} // namespace tidy
} // namespace clang
