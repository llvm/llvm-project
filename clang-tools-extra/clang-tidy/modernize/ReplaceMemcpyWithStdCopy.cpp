//===--- ReplaceMemcpyWithStdCopy.cpp - clang-tidy----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReplaceMemcpyWithStdCopy.h"
#include "../utils/Matchers.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Tooling/FixIt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <array>
#include <optional>
#include <type_traits>
#include <variant>

using namespace clang;
using namespace clang::ast_matchers;

namespace clang::tidy::modernize {
static constexpr llvm::StringLiteral MemcpyRef = "::std::memcpy";

namespace {
// Helper Matcher which applies the given QualType Matcher either directly or by
// resolving a pointer type to its pointee. Used to match v.push_back() as well
// as p->push_back().
auto hasTypeOrPointeeType(
    const ast_matchers::internal::Matcher<QualType> &TypeMatcher) {
  return anyOf(hasType(TypeMatcher),
               hasType(pointerType(pointee(TypeMatcher))));
}

constexpr llvm::StringLiteral StdCopyNHeader = "<algorithm>";
} // namespace

ReplaceMemcpyWithStdCopy::ReplaceMemcpyWithStdCopy(StringRef Name,
                                                   ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM),
               areDiagsSelfContained()) {}

void ReplaceMemcpyWithStdCopy::registerMatchers(MatchFinder *Finder) {
  const auto ByteLikeTypeWithId = [](const char *ID) {
    return anything();
    // return hasCanonicalType(matchers::isSimpleChar());
  };

  const auto IsByteSequence = anything();
  // hasTypeOrPointeeType(
  //     hasCanonicalType(hasDeclaration(cxxRecordDecl(hasAnyName({
  //         "::std::vector",
  //         "::std::span",
  //         "::std::deque",
  //         "::std::array",
  //         "::std::string",
  //         "::std::string_view",
  //     })))));
  // __jm__ for template classes need to add a check whether T is ByteLike
  // __jm__ these matchers unused because memcpy for non-byte sequences should
  // be flagged without FixIt

  const auto IsDestContainerData = cxxMemberCallExpr(
      callee(cxxMethodDecl(
          hasName("data"),
          returns(pointerType(pointee(ByteLikeTypeWithId("data_return_type")))),
          unless(isConst()))),
      argumentCountIs(0), on(expr(IsByteSequence)));

  const auto IsSourceContainerData = cxxMemberCallExpr(
      callee(cxxMethodDecl(
          hasName("data"),
          returns(pointerType(pointee(ByteLikeTypeWithId("data_return_type"))))
          //,isConst() __jm__ doesn't have to be const pre-implicit cast, but
          // that would require switching traversal type
          )),
      argumentCountIs(0),
      on(expr(IsByteSequence))); // __jm__ how to express that the caller needs
                                 // to be const qualified / a const lvalue?

  auto IsDestCArrayOrContainerData =
      anyOf(IsDestContainerData.bind("dest_data"),
            hasType(hasCanonicalType(arrayType().bind("dest_array"))));

  auto IsSourceCArrayOrContainerData =
      anyOf(IsSourceContainerData, hasType(hasCanonicalType(arrayType())));
  // all of the pointer args to memcpy must be any of:
  // 1. .data() call to record which
  // can be passed to std::begin or has .begin method and
  // can be passed to std::end   or has .end   method and
  // can be passed to std::size  or has .size  method
  // 2. static c-array
  const auto ReturnValueUsed =
      optionally(hasParent(compoundStmt().bind("return_value_discarded")));

  const auto MemcpyDecl =
      functionDecl(hasAnyName("::std::memcpy", "::memcpy")
                   // ,             argumentCountIs(3) __jm__ doesn't work for
                   // functionDecl, possibly only for callExpr?
      );
  // __jm__ also match on arg types
  const auto Expression =
      callExpr(callee(MemcpyDecl), ReturnValueUsed,
               hasArgument(0, expr(IsDestCArrayOrContainerData).bind("dest")),
               hasArgument(1, IsSourceCArrayOrContainerData));
  Finder->addMatcher(Expression.bind(MemcpyRef), this);
}

void ReplaceMemcpyWithStdCopy::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void ReplaceMemcpyWithStdCopy::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
}

// Determine if the result of an expression is "stored" in some way.
// It is true if the value is stored into a variable or used as initialization
// or passed to a function or constructor.
// For this use case compound assignments are not counted as a "store" (the 'E'
// expression should have pointer type).
static bool isExprValueStored(const Expr *E, ASTContext &C) {
  E = E->IgnoreParenCasts();
  // Get first non-paren, non-cast parent.
  ParentMapContext &PMap = C.getParentMapContext();
  DynTypedNodeList P = PMap.getParents(*E);
  if (P.size() != 1)
    return false;
  const Expr *ParentE = nullptr;
  while ((ParentE = P[0].get<Expr>()) && ParentE->IgnoreParenCasts() == E) {
    P = PMap.getParents(P[0]);
    if (P.size() != 1)
      return false;
  }

  if (const auto *ParentVarD = P[0].get<VarDecl>())
    return ParentVarD->getInit()->IgnoreParenCasts() == E;

  if (!ParentE)
    return false;

  if (const auto *BinOp = dyn_cast<BinaryOperator>(ParentE))
    return BinOp->getOpcode() == BO_Assign &&
           BinOp->getRHS()->IgnoreParenCasts() == E;

  return isa<CallExpr, CXXConstructExpr>(ParentE);
}
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "clang-tidy"

bool ReplaceMemcpyWithStdCopy::checkIsByteSequence(
    const MatchFinder::MatchResult &Result, std::string_view Prefix) {

  const auto &ArgNode = *Result.Nodes.getNodeAs<Expr>("dest");
  ArgNode.dumpColor();

  auto PointerType_ = ArgNode.getType()->getAs<PointerType>();
  if (PointerType_ == nullptr)
    return false;

  auto PointeeType = PointerType_->getPointeeType();
  // if
  // (StructFieldTy->isIncompleteType())
  //   return false;
  auto PointeeWidth =
      Result.Context->getTypeInfo(PointeeType.getTypePtr()).Width;

  bool IsArgPtrToBytelike = PointeeWidth == Result.Context->getCharWidth();

  LLVM_DEBUG(llvm::dbgs() << "__jm__ dbgs\n");
  // ArgNode.dumpPretty(*Result.Context);
  ArgNode.dumpColor();
  if (ArgNode.IgnoreImplicit()->getType()->isArrayType()) {
    LLVM_DEBUG(llvm::dbgs()
               << name<CArray>() << "__jm__ array \n typewidth:" << PointeeWidth
               << "\n charwidth:" << Result.Context->getCharWidth());

    return IsArgPtrToBytelike;
  }

  // ArgNode is a result of either std::data or .data()
  if (const auto *MC =
          dyn_cast_or_null<CXXMemberCallExpr>(ArgNode.IgnoreCasts());
      MC != nullptr) {
    // return false;
    LLVM_DEBUG(llvm::dbgs() << "__jm__ member call");
    // diag(ArgNode.getExprLoc(), "%0 isnull")
    //      << (MC->getMethodDecl() == nullptr);
  } else {
    LLVM_DEBUG(llvm::dbgs() << "__jm__ otherwise");
  }
  return false;
  //  else if (auto *FC = P->get<CallExpr>(); FC != nullptr) {
  //   diag(ArgNode.getExprLoc(), "%0 _2 name is")
  //        <<
  //        FC->getCalleeDecl()->getCanonicalDecl()->getAsFunction()->getName();
  // }
}

void ReplaceMemcpyWithStdCopy::check(const MatchFinder::MatchResult &Result) {
  const auto *MemcpyNode = Result.Nodes.getNodeAs<CallExpr>(MemcpyRef);
  assert(MemcpyNode != nullptr && "Matched node cannot be null");

  auto Diag = diag(MemcpyNode->getExprLoc(), "prefer std::copy_n to memcpy");

  // don't issue a fixit if the result of the call is used
  // if (bool IsMemcpyReturnValueUsed =
  //         Result.Nodes.getNodeAs<Expr>("return_value_discarded") == nullptr;
  //     IsMemcpyReturnValueUsed)
  //   return;

  auto *DestArg = MemcpyNode->getArg(0);
  auto *SrcArg = MemcpyNode->getArg(1);
  auto *SizeArg = MemcpyNode->getArg(2);

  assert(DestArg != nullptr and SrcArg != nullptr and SizeArg != nullptr and
         "Call node arguments cannot be null");

  if (not checkIsByteSequence(Result, "dest")
      // or
      //     not checkIsByteSequence(Result, *SrcArg, Diag)
  )
    diag(MemcpyNode->getExprLoc(), "not a byte sequence");

  // // __jm__ strip .data() / std::data calls

  // // const CharSourceRange FunctionNameSourceRange =
  // // CharSourceRange::getCharRange(
  // //     MemcpyNode->getBeginLoc(), MemcpyNode->getArg(0)->getBeginLoc());
  // Diag << FixItHint::CreateReplacement(
  //     MemcpyNode->getCallee()->getSourceRange(), "std::copy_n");

  {
    // using namespace std::literals;
    // Diag << FixItHint::CreateReplacement(
    //     DestArg->getSourceRange(),
    //     ("std::begin(" +
    //      tooling::fixit::getText(SrcArg->getSourceRange(), *Result.Context) +
    //      ")")
    //         .str());
    // Diag << FixItHint::CreateReplacement(
    //     SrcArg->getSourceRange(),
    //     tooling::fixit::getText(SizeArg->getSourceRange(), *Result.Context));
    // Diag << FixItHint::CreateReplacement(
    //     SizeArg->getSourceRange(),
    //     ("std::begin(" +
    //      tooling::fixit::getText(DestArg->getSourceRange(), *Result.Context)
    //      +
    //      ")")
    //         .str());
  }
  // dest source size -> source size dest
  // renameFunction(Diag, MemcpyNode);
  // reorderArgs(Diag, MemcpyNode);
  // insertHeader(Diag, MemcpyNode, Result.SourceManager);

  Diag << Inserter.createIncludeInsertion(
      Result.SourceManager->getFileID(MemcpyNode->getBeginLoc()),
      StdCopyNHeader);
}

// void ReplaceMemcpyWithStdCopy::handleMemcpy(const CallExpr &Node) {

//   // FixIt
//   const CharSourceRange FunctionNameSourceRange =
//   CharSourceRange::getCharRange(
//       Node.getBeginLoc(), Node.getArg(0)->getBeginLoc());

//   Diag << FixItHint::CreateReplacement(FunctionNameSourceRange,
//   "std::copy_n(");
// }

void ReplaceMemcpyWithStdCopy::renameFunction(DiagnosticBuilder &Diag,
                                              const CallExpr *MemcpyNode) {}

void ReplaceMemcpyWithStdCopy::reorderArgs(DiagnosticBuilder &Diag,
                                           const CallExpr *MemcpyNode) {
  std::array<std::string, 3> Arg;

  LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  PrintingPolicy Policy(LangOpts);

  // Retrieve all the arguments
  for (uint8_t I = 0; I < Arg.size(); I++) {
    llvm::raw_string_ostream S(Arg[I]);
    MemcpyNode->getArg(I)->printPretty(S, nullptr, Policy);
  }

  // Create lambda that return SourceRange of an argument
  auto GetSourceRange = [MemcpyNode](uint8_t ArgCount) -> SourceRange {
    return SourceRange(MemcpyNode->getArg(ArgCount)->getBeginLoc(),
                       MemcpyNode->getArg(ArgCount)->getEndLoc());
  };

  // Reorder the arguments
  Diag << FixItHint::CreateReplacement(GetSourceRange(0), Arg[1]);

  Arg[2] = Arg[1] + " + ((" + Arg[2] + ") / sizeof(*(" + Arg[1] + ")))";
  Diag << FixItHint::CreateReplacement(GetSourceRange(1), Arg[2]);

  Diag << FixItHint::CreateReplacement(GetSourceRange(2), Arg[0]);
}
} // namespace clang::tidy::modernize
