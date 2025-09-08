//===--- MakeSmartPtrCheck.cpp - clang-tidy--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../utils/TypeTraits.h"
#include "MakeSharedCheck.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

static constexpr char ConstructorCall[] = "constructorCall";
static constexpr char DirectVar[] = "directVar";
static constexpr char ResetCall[] = "resetCall";
static constexpr char NewExpression[] = "newExpression";

static std::string getNewExprName(const CXXNewExpr *NewExpr,
                                  const SourceManager &SM,
                                  const LangOptions &Lang) {
  StringRef WrittenName = Lexer::getSourceText(
      CharSourceRange::getTokenRange(
          NewExpr->getAllocatedTypeSourceInfo()->getTypeLoc().getSourceRange()),
      SM, Lang);
  if (NewExpr->isArray()) {
    return (WrittenName + "[]").str();
  }
  return WrittenName.str();
}

const char MakeSmartPtrCheck::PointerType[] = "pointerType";

MakeSmartPtrCheck::MakeSmartPtrCheck(StringRef Name, ClangTidyContext *Context,
                                     StringRef MakeSmartPtrFunctionName)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM),
               areDiagsSelfContained()),
      MakeSmartPtrFunctionHeader(
          Options.get("MakeSmartPtrFunctionHeader", "<memory>")),
      MakeSmartPtrFunctionName(
          Options.get("MakeSmartPtrFunction", MakeSmartPtrFunctionName)),
      IgnoreMacros(Options.get("IgnoreMacros", true)),
      IgnoreDefaultInitialization(
          Options.get("IgnoreDefaultInitialization", true)) {}

void MakeSmartPtrCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
  Options.store(Opts, "MakeSmartPtrFunctionHeader", MakeSmartPtrFunctionHeader);
  Options.store(Opts, "MakeSmartPtrFunction", MakeSmartPtrFunctionName);
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
  Options.store(Opts, "IgnoreDefaultInitialization",
                IgnoreDefaultInitialization);
}

bool MakeSmartPtrCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus11;
}

void MakeSmartPtrCheck::registerPPCallbacks(const SourceManager &SM,
                                            Preprocessor *PP,
                                            Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void MakeSmartPtrCheck::registerMatchers(ast_matchers::MatchFinder *Finder) {
  // Calling make_smart_ptr from within a member function of a type with a
  // private or protected constructor would be ill-formed.
  auto CanCallCtor = unless(has(ignoringImpCasts(
      cxxConstructExpr(hasDeclaration(decl(unless(isPublic())))))));

  auto IsPlacement = hasAnyPlacementArg(anything());

  Finder->addMatcher(
      traverse(TK_AsIs,
               cxxConstructExpr(
                   anyOf(hasParent(cxxBindTemporaryExpr()),
                         hasParent(varDecl().bind(DirectVar))),
                   hasType(getSmartPointerTypeMatcher()), argumentCountIs(1),
                   hasArgument(
                       0, cxxNewExpr(hasType(pointsTo(qualType(hasCanonicalType(
                                         equalsBoundNode(PointerType))))),
                                     CanCallCtor, unless(IsPlacement))
                              .bind(NewExpression)),
                   unless(isInTemplateInstantiation()))
                   .bind(ConstructorCall)),
      this);

  Finder->addMatcher(
      traverse(
          TK_AsIs,
          cxxMemberCallExpr(
              unless(isInTemplateInstantiation()),
              hasArgument(0, cxxNewExpr(CanCallCtor, unless(IsPlacement))
                                 .bind(NewExpression)),
              callee(cxxMethodDecl(hasName("reset"))),
              anyOf(thisPointerType(getSmartPointerTypeMatcher()),
                    on(ignoringImplicit(anyOf(
                        hasType(getSmartPointerTypeMatcher()),
                        hasType(pointsTo(getSmartPointerTypeMatcher())))))))
              .bind(ResetCall)),
      this);
}

void MakeSmartPtrCheck::check(const MatchFinder::MatchResult &Result) {
  // 'smart_ptr' refers to 'std::shared_ptr' or 'std::unique_ptr' or other
  // pointer, 'make_smart_ptr' refers to 'std::make_shared' or
  // 'std::make_unique' or other function that creates smart_ptr.

  SourceManager &SM = *Result.SourceManager;
  const auto *Construct =
      Result.Nodes.getNodeAs<CXXConstructExpr>(ConstructorCall);
  const auto *DVar = Result.Nodes.getNodeAs<VarDecl>(DirectVar);
  const auto *Reset = Result.Nodes.getNodeAs<CXXMemberCallExpr>(ResetCall);
  const auto *Type = Result.Nodes.getNodeAs<QualType>(PointerType);
  const auto *New = Result.Nodes.getNodeAs<CXXNewExpr>(NewExpression);

  // Skip when this is a new-expression with `auto`, e.g. new auto(1)
  if (New->getType()->getPointeeType()->getContainedAutoType())
    return;

  // Be conservative for cases where we construct and default initialize.
  //
  // For example,
  //    P.reset(new int)    // check fix: P = std::make_unique<int>()
  //    P.reset(new int[5]) // check fix: P = std::make_unique<int []>(5)
  //
  // The fix of the check has side effect, it introduces value initialization
  // which maybe unexpected and cause performance regression.
  bool Initializes = New->hasInitializer() ||
                     !utils::type_traits::isTriviallyDefaultConstructible(
                         New->getAllocatedType(), *Result.Context);
  if (!Initializes && IgnoreDefaultInitialization)
    return;
  if (Construct)
    checkConstruct(SM, Result.Context, Construct, DVar, Type, New);
  else if (Reset)
    checkReset(SM, Result.Context, Reset, New);
}

void MakeSmartPtrCheck::checkConstruct(SourceManager &SM, ASTContext *Ctx,
                                       const CXXConstructExpr *Construct,
                                       const VarDecl *DVar,
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

  if (!replaceNew(Diag, New, SM, Ctx)) {
    return;
  }

  // Find the location of the template's left angle.
  size_t LAngle = ExprStr.find('<');
  SourceLocation ConstructCallEnd;
  if (LAngle == StringRef::npos) {
    // If the template argument is missing (because it is part of the alias)
    // we have to add it back.
    ConstructCallEnd = ConstructCallStart.getLocWithOffset(ExprStr.size());
    Diag << FixItHint::CreateInsertion(
        ConstructCallEnd, "<" + getNewExprName(New, SM, getLangOpts()) + ">");
  } else {
    ConstructCallEnd = ConstructCallStart.getLocWithOffset(LAngle);
  }

  std::string FinalMakeSmartPtrFunctionName = MakeSmartPtrFunctionName.str();
  if (DVar)
    FinalMakeSmartPtrFunctionName =
        ExprStr.str() + " = " + MakeSmartPtrFunctionName.str();

  Diag << FixItHint::CreateReplacement(
      CharSourceRange::getCharRange(ConstructCallStart, ConstructCallEnd),
      FinalMakeSmartPtrFunctionName);

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

void MakeSmartPtrCheck::checkReset(SourceManager &SM, ASTContext *Ctx,
                                   const CXXMemberCallExpr *Reset,
                                   const CXXNewExpr *New) {
  const auto *Expr = cast<MemberExpr>(Reset->getCallee());
  SourceLocation OperatorLoc = Expr->getOperatorLoc();
  SourceLocation ResetCallStart = Reset->getExprLoc();
  SourceLocation ExprStart = Expr->getBeginLoc();
  SourceLocation ExprEnd =
      Lexer::getLocForEndOfToken(Expr->getEndLoc(), 0, SM, getLangOpts());

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

  if (!replaceNew(Diag, New, SM, Ctx)) {
    return;
  }

  Diag << FixItHint::CreateReplacement(
      CharSourceRange::getCharRange(OperatorLoc, ExprEnd),
      (llvm::Twine(" = ") + MakeSmartPtrFunctionName + "<" +
       getNewExprName(New, SM, getLangOpts()) + ">")
          .str());

  if (Expr->isArrow())
    Diag << FixItHint::CreateInsertion(ExprStart, "*");

  insertHeader(Diag, SM.getFileID(OperatorLoc));
}

bool MakeSmartPtrCheck::replaceNew(DiagnosticBuilder &Diag,
                                   const CXXNewExpr *New, SourceManager &SM,
                                   ASTContext *Ctx) {
  auto SkipParensParents = [&](const Expr *E) {
    TraversalKindScope RAII(*Ctx, TK_AsIs);

    for (const Expr *OldE = nullptr; E != OldE;) {
      OldE = E;
      for (const auto &Node : Ctx->getParents(*E)) {
        if (const Expr *Parent = Node.get<ParenExpr>()) {
          E = Parent;
          break;
        }
      }
    }
    return E;
  };

  SourceRange NewRange = SkipParensParents(New)->getSourceRange();
  SourceLocation NewStart = NewRange.getBegin();
  SourceLocation NewEnd = NewRange.getEnd();

  // Skip when the source location of the new expression is invalid.
  if (NewStart.isInvalid() || NewEnd.isInvalid())
    return false;

  std::string ArraySizeExpr;
  if (const auto *ArraySize = New->getArraySize().value_or(nullptr)) {
    ArraySizeExpr = Lexer::getSourceText(CharSourceRange::getTokenRange(
                                             ArraySize->getSourceRange()),
                                         SM, getLangOpts())
                        .str();
  }
  // Returns true if the given constructor expression has any braced-init-list
  // argument, e.g.
  //   Foo({1, 2}, 1) => true
  //   Foo(Bar{1, 2}) => true
  //   Foo(1) => false
  //   Foo{1} => false
  auto HasListIntializedArgument = [](const CXXConstructExpr *CE) {
    for (const auto *Arg : CE->arguments()) {
      Arg = Arg->IgnoreImplicit();

      if (isa<CXXStdInitializerListExpr>(Arg) || isa<InitListExpr>(Arg))
        return true;
      // Check whether we implicitly construct a class from a
      // std::initializer_list.
      if (const auto *CEArg = dyn_cast<CXXConstructExpr>(Arg)) {
        // Strip the elidable move constructor, it is present in the AST for
        // C++11/14, e.g. Foo(Bar{1, 2}), the move constructor is around the
        // init-list constructor.
        if (CEArg->isElidable()) {
          if (const auto *TempExp = CEArg->getArg(0)) {
            if (const auto *UnwrappedCE =
                    dyn_cast<CXXConstructExpr>(TempExp->IgnoreImplicit()))
              CEArg = UnwrappedCE;
          }
        }
        if (CEArg->isStdInitListInitialization())
          return true;
      }
    }
    return false;
  };
  switch (New->getInitializationStyle()) {
  case CXXNewInitializationStyle::None: {
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
  case CXXNewInitializationStyle::Parens: {
    // FIXME: Add fixes for constructors with parameters that can be created
    // with a C++11 braced-init-list (e.g. std::vector, std::map).
    // Unlike ordinal cases, braced list can not be deduced in
    // std::make_smart_ptr, we need to specify the type explicitly in the fixes:
    //   struct S { S(std::initializer_list<int>, int); };
    //   struct S2 { S2(std::vector<int>); };
    //   struct S3 { S3(S2, int); };
    //   smart_ptr<S>(new S({1, 2, 3}, 1));  // C++98 call-style initialization
    //   smart_ptr<S>(new S({}, 1));
    //   smart_ptr<S2>(new S2({1})); // implicit conversion:
    //                               //   std::initializer_list => std::vector
    //   smart_ptr<S3>(new S3({1, 2}, 3));
    // The above samples have to be replaced with:
    //   std::make_smart_ptr<S>(std::initializer_list<int>({1, 2, 3}), 1);
    //   std::make_smart_ptr<S>(std::initializer_list<int>({}), 1);
    //   std::make_smart_ptr<S2>(std::vector<int>({1}));
    //   std::make_smart_ptr<S3>(S2{1, 2}, 3);
    if (const auto *CE = New->getConstructExpr()) {
      if (HasListIntializedArgument(CE))
        return false;
    }
    if (ArraySizeExpr.empty()) {
      SourceRange InitRange = New->getDirectInitRange();
      Diag << FixItHint::CreateRemoval(
          SourceRange(NewStart, InitRange.getBegin()));
      Diag << FixItHint::CreateRemoval(SourceRange(InitRange.getEnd(), NewEnd));
    } else {
      // New array expression with default/value initialization:
      //   smart_ptr<Foo[]>(new int[5]());
      //   smart_ptr<Foo[]>(new Foo[5]());
      Diag << FixItHint::CreateReplacement(SourceRange(NewStart, NewEnd),
                                           ArraySizeExpr);
    }
    break;
  }
  case CXXNewInitializationStyle::Braces: {
    // Range of the substring that we do not want to remove.
    SourceRange InitRange;
    if (const auto *NewConstruct = New->getConstructExpr()) {
      if (NewConstruct->isStdInitListInitialization() ||
          HasListIntializedArgument(NewConstruct)) {
        // FIXME: Add fixes for direct initialization with the initializer-list
        // constructor. Similar to the above CallInit case, the type has to be
        // specified explicitly in the fixes.
        //   struct S { S(std::initializer_list<int>); };
        //   struct S2 { S2(S, int); };
        //   smart_ptr<S>(new S{1, 2, 3});  // C++11 direct list-initialization
        //   smart_ptr<S>(new S{});  // use initializer-list constructor
        //   smart_ptr<S2>()new S2{ {1,2}, 3 }; // have a list-initialized arg
        // The above cases have to be replaced with:
        //   std::make_smart_ptr<S>(std::initializer_list<int>({1, 2, 3}));
        //   std::make_smart_ptr<S>(std::initializer_list<int>({}));
        //   std::make_smart_ptr<S2>(S{1, 2}, 3);
        return false;
      }
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
    } else {
      // Aggregate initialization.
      //   smart_ptr<Pair>(new Pair{first, second});
      // Has to be replaced with:
      //   smart_ptr<Pair>(Pair{first, second});
      //
      // The fix (std::make_unique) needs to see copy/move constructor of
      // Pair. If we found any invisible or deleted copy/move constructor, we
      // stop generating fixes -- as the C++ rule is complicated and we are less
      // certain about the correct fixes.
      if (const CXXRecordDecl *RD = New->getType()->getPointeeCXXRecordDecl()) {
        if (llvm::any_of(RD->ctors(), [](const CXXConstructorDecl *Ctor) {
              return Ctor->isCopyOrMoveConstructor() &&
                     (Ctor->isDeleted() || Ctor->getAccess() == AS_private);
            })) {
          return false;
        }
      }
      InitRange = SourceRange(
          New->getAllocatedTypeSourceInfo()->getTypeLoc().getBeginLoc(),
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
  Diag << Inserter.createIncludeInsertion(FD, MakeSmartPtrFunctionHeader);
}

} // namespace clang::tidy::modernize
