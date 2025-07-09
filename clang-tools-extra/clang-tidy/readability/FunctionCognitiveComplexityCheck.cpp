//===--- FunctionCognitiveComplexityCheck.cpp - clang-tidy ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FunctionCognitiveComplexityCheck.h"
#include "../ClangTidyDiagnosticConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/ErrorHandling.h"
#include <array>
#include <cassert>
#include <optional>
#include <stack>
#include <tuple>
#include <utility>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {
namespace {

struct CognitiveComplexity final {
  // Any increment is based on some combination of reasons.
  // For details you can look at the Specification at
  // https://www.sonarsource.com/docs/CognitiveComplexity.pdf
  // or user-facing docs at
  // http://clang.llvm.org/extra/clang-tidy/checks/readability/function-cognitive-complexity.html
  // Here are all the possible reasons:
  enum Criteria : uint8_t {
    None = 0U,

    // B1, increases cognitive complexity (by 1)
    // What causes it:
    // * if, else if, else, ConditionalOperator (not BinaryConditionalOperator)
    // * SwitchStmt
    // * ForStmt, CXXForRangeStmt
    // * WhileStmt, DoStmt
    // * CXXCatchStmt
    // * GotoStmt, IndirectGotoStmt (but not BreakStmt, ContinueStmt)
    // * sequences of binary logical operators (BinOpLAnd, BinOpLOr)
    // * each method in a recursion cycle (not implemented)
    Increment = 1U << 0,

    // B2, increases current nesting level (by 1)
    // What causes it:
    // * if, else if, else, ConditionalOperator (not BinaryConditionalOperator)
    // * SwitchStmt
    // * ForStmt, CXXForRangeStmt
    // * WhileStmt, DoStmt
    // * CXXCatchStmt
    // * nested CXXConstructor, CXXDestructor, CXXMethod (incl. C++11 Lambda)
    // * GNU Statement Expression
    // * Apple Block declaration
    IncrementNesting = 1U << 1,

    // B3, increases cognitive complexity by the current nesting level
    // Applied before IncrementNesting
    // What causes it:
    // * IfStmt, ConditionalOperator (not BinaryConditionalOperator)
    // * SwitchStmt
    // * ForStmt, CXXForRangeStmt
    // * WhileStmt, DoStmt
    // * CXXCatchStmt
    PenalizeNesting = 1U << 2,

    All = Increment | PenalizeNesting | IncrementNesting,
  };

  // The helper struct used to record one increment occurrence, with all the
  // details necessary.
  struct Detail {
    const SourceLocation Loc;     // What caused the increment?
    const unsigned short Nesting; // How deeply nested is Loc located?
    const Criteria C;             // The criteria of the increment

    Detail(SourceLocation SLoc, unsigned short CurrentNesting, Criteria Crit)
        : Loc(SLoc), Nesting(CurrentNesting), C(Crit) {}

    // To minimize the sizeof(Detail), we only store the minimal info there.
    // This function is used to convert from the stored info into the usable
    // information - what message to output, how much of an increment did this
    // occurrence actually result in.
    std::pair<unsigned, unsigned short> process() const {
      assert(C != Criteria::None && "invalid criteria");

      unsigned MsgId = 0;           // The id of the message to output.
      unsigned short Increment = 0; // How much of an increment?

      if (C == Criteria::All) {
        Increment = 1 + Nesting;
        MsgId = 0;
      } else if (C == (Criteria::Increment | Criteria::IncrementNesting)) {
        Increment = 1;
        MsgId = 1;
      } else if (C == Criteria::Increment) {
        Increment = 1;
        MsgId = 2;
      } else if (C == Criteria::IncrementNesting) {
        Increment = 0; // Unused in this message.
        MsgId = 3;
      } else
        llvm_unreachable("should not get to here.");

      return std::make_pair(MsgId, Increment);
    }
  };

  // Limit of 25 is the "upstream"'s default.
  static constexpr unsigned DefaultLimit = 25U;

  // Based on the publicly-available numbers for some big open-source projects
  // https://sonarcloud.io/projects?languages=c%2Ccpp&size=5   we can estimate:
  // value ~20 would result in no allocs for 98% of functions, ~12 for 96%, ~10
  // for 91%, ~8 for 88%, ~6 for 84%, ~4 for 77%, ~2 for 64%, and ~1 for 37%.
  static_assert(sizeof(Detail) <= 8,
                "Since we use SmallVector to minimize the amount of "
                "allocations, we also need to consider the price we pay for "
                "that in terms of stack usage. "
                "Thus, it is good to minimize the size of the Detail struct.");
  SmallVector<Detail, DefaultLimit> Details; // 25 elements is 200 bytes.
  // Yes, 25 is a magic number. This is the seemingly-sane default for the
  // upper limit for function cognitive complexity. Thus it would make sense
  // to avoid allocations for any function that does not violate the limit.

  // The grand total Cognitive Complexity of the function.
  unsigned Total = 0;

  // The function used to store new increment, calculate the total complexity.
  void account(SourceLocation Loc, unsigned short Nesting, Criteria C);
};

// All the possible messages that can be output. The choice of the message
// to use is based of the combination of the CognitiveComplexity::Criteria.
// It would be nice to have it in CognitiveComplexity struct, but then it is
// not static.
static const std::array<const StringRef, 4> Msgs = {{
    // B1 + B2 + B3
    "+%0, including nesting penalty of %1, nesting level increased to %2",

    // B1 + B2
    "+%0, nesting level increased to %2",

    // B1
    "+%0",

    // B2
    "nesting level increased to %2",
}};

// Criteria is a bitset, thus a few helpers are needed.
CognitiveComplexity::Criteria operator|(CognitiveComplexity::Criteria LHS,
                                        CognitiveComplexity::Criteria RHS) {
  return static_cast<CognitiveComplexity::Criteria>(llvm::to_underlying(LHS) |
                                                    llvm::to_underlying(RHS));
}
CognitiveComplexity::Criteria operator&(CognitiveComplexity::Criteria LHS,
                                        CognitiveComplexity::Criteria RHS) {
  return static_cast<CognitiveComplexity::Criteria>(llvm::to_underlying(LHS) &
                                                    llvm::to_underlying(RHS));
}
CognitiveComplexity::Criteria &operator|=(CognitiveComplexity::Criteria &LHS,
                                          CognitiveComplexity::Criteria RHS) {
  LHS = operator|(LHS, RHS);
  return LHS;
}
CognitiveComplexity::Criteria &operator&=(CognitiveComplexity::Criteria &LHS,
                                          CognitiveComplexity::Criteria RHS) {
  LHS = operator&(LHS, RHS);
  return LHS;
}

void CognitiveComplexity::account(SourceLocation Loc, unsigned short Nesting,
                                  Criteria C) {
  C &= Criteria::All;
  assert(C != Criteria::None && "invalid criteria");

  Details.emplace_back(Loc, Nesting, C);
  const Detail &D = Details.back();

  unsigned MsgId = 0;
  unsigned short Increase = 0;
  std::tie(MsgId, Increase) = D.process();

  Total += Increase;
}

class FunctionASTVisitor final
    : public RecursiveASTVisitor<FunctionASTVisitor> {
  using Base = RecursiveASTVisitor<FunctionASTVisitor>;

  // If set to true, macros are ignored during analysis.
  const bool IgnoreMacros;

  // The current nesting level (increased by Criteria::IncrementNesting).
  unsigned short CurrentNestingLevel = 0;

  // Used to efficiently know the last type of the binary sequence operator
  // that was encountered. It would make sense for the function call to start
  // the new sequence, thus it is a stack.
  using OBO = std::optional<BinaryOperator::Opcode>;
  std::stack<OBO, SmallVector<OBO, 4>> BinaryOperatorsStack;

public:
  explicit FunctionASTVisitor(const bool IgnoreMacros)
      : IgnoreMacros(IgnoreMacros) {}

  bool traverseStmtWithIncreasedNestingLevel(Stmt *Node) {
    ++CurrentNestingLevel;
    bool ShouldContinue = Base::TraverseStmt(Node);
    --CurrentNestingLevel;
    return ShouldContinue;
  }

  bool traverseDeclWithIncreasedNestingLevel(Decl *Node) {
    ++CurrentNestingLevel;
    bool ShouldContinue = Base::TraverseDecl(Node);
    --CurrentNestingLevel;
    return ShouldContinue;
  }

  bool TraverseIfStmt(IfStmt *Node, bool InElseIf = false) {
    if (!Node)
      return Base::TraverseIfStmt(Node);

    {
      CognitiveComplexity::Criteria Reasons =
          CognitiveComplexity::Criteria::None;

      // "If" increases cognitive complexity.
      Reasons |= CognitiveComplexity::Criteria::Increment;
      // "If" increases nesting level.
      Reasons |= CognitiveComplexity::Criteria::IncrementNesting;

      if (!InElseIf) {
        // "If" receives a nesting increment commensurate with it's nested
        // depth, if it is not part of "else if".
        Reasons |= CognitiveComplexity::Criteria::PenalizeNesting;
      }

      CC.account(Node->getIfLoc(), CurrentNestingLevel, Reasons);
    }

    // If this IfStmt is *NOT* "else if", then only the body (i.e. "Then" and
    // "Else") is traversed with increased Nesting level.
    // However if this IfStmt *IS* "else if", then Nesting level is increased
    // for the whole IfStmt (i.e. for "Init", "Cond", "Then" and "Else").

    if (!InElseIf) {
      if (!TraverseStmt(Node->getInit()))
        return false;

      if (!TraverseStmt(Node->getCond()))
        return false;
    } else {
      if (!traverseStmtWithIncreasedNestingLevel(Node->getInit()))
        return false;

      if (!traverseStmtWithIncreasedNestingLevel(Node->getCond()))
        return false;
    }

    // "Then" always increases nesting level.
    if (!traverseStmtWithIncreasedNestingLevel(Node->getThen()))
      return false;

    if (!Node->getElse())
      return true;

    if (auto *E = dyn_cast<IfStmt>(Node->getElse()))
      return TraverseIfStmt(E, true);

    {
      CognitiveComplexity::Criteria Reasons =
          CognitiveComplexity::Criteria::None;

      // "Else" increases cognitive complexity.
      Reasons |= CognitiveComplexity::Criteria::Increment;
      // "Else" increases nesting level.
      Reasons |= CognitiveComplexity::Criteria::IncrementNesting;
      // "Else" DOES NOT receive a nesting increment commensurate with it's
      // nested depth.

      CC.account(Node->getElseLoc(), CurrentNestingLevel, Reasons);
    }

    // "Else" always increases nesting level.
    return traverseStmtWithIncreasedNestingLevel(Node->getElse());
  }

// The currently-being-processed stack entry, which is always the top.
#define CurrentBinaryOperator BinaryOperatorsStack.top()

  // In a sequence of binary logical operators, if the new operator is different
  // from the previous one, then the cognitive complexity is increased.
  bool TraverseBinaryOperator(BinaryOperator *Op) {
    if (!Op || !Op->isLogicalOp())
      return Base::TraverseBinaryOperator(Op);

    // Make sure that there is always at least one frame in the stack.
    if (BinaryOperatorsStack.empty())
      BinaryOperatorsStack.emplace();

    // If this is the first binary operator that we are processing, or the
    // previous binary operator was different, there is an increment.
    if (!CurrentBinaryOperator || Op->getOpcode() != CurrentBinaryOperator)
      CC.account(Op->getOperatorLoc(), CurrentNestingLevel,
                 CognitiveComplexity::Criteria::Increment);

    // We might encounter a function call, which starts a new sequence, thus
    // we need to save the current previous binary operator.
    const std::optional<BinaryOperator::Opcode> BinOpCopy(
        CurrentBinaryOperator);

    // Record the operator that we are currently processing and traverse it.
    CurrentBinaryOperator = Op->getOpcode();
    bool ShouldContinue = Base::TraverseBinaryOperator(Op);

    // And restore the previous binary operator, which might be nonexistent.
    CurrentBinaryOperator = BinOpCopy;

    return ShouldContinue;
  }

  // It would make sense for the function call to start the new binary
  // operator sequence, thus let's make sure that it creates a new stack frame.
  bool TraverseCallExpr(CallExpr *Node) {
    // If we are not currently processing any binary operator sequence, then
    // no Node-handling is needed.
    if (!Node || BinaryOperatorsStack.empty() || !CurrentBinaryOperator)
      return Base::TraverseCallExpr(Node);

    // Else, do add [uninitialized] frame to the stack, and traverse call.
    BinaryOperatorsStack.emplace();
    bool ShouldContinue = Base::TraverseCallExpr(Node);
    // And remove the top frame.
    BinaryOperatorsStack.pop();

    return ShouldContinue;
  }

#undef CurrentBinaryOperator

  bool TraverseStmt(Stmt *Node) {
    if (!Node)
      return Base::TraverseStmt(Node);

    if (IgnoreMacros && Node->getBeginLoc().isMacroID())
      return true;

    // Three following switch()'es have huge duplication, but it is better to
    // keep them separate, to simplify comparing them with the Specification.

    CognitiveComplexity::Criteria Reasons = CognitiveComplexity::Criteria::None;
    SourceLocation Location = Node->getBeginLoc();

    // B1. Increments
    // There is an increment for each of the following:
    switch (Node->getStmtClass()) {
    // if, else if, else are handled in TraverseIfStmt(),
    // FIXME: "each method in a recursion cycle" Increment is not implemented.
    case Stmt::ConditionalOperatorClass:
    case Stmt::SwitchStmtClass:
    case Stmt::ForStmtClass:
    case Stmt::CXXForRangeStmtClass:
    case Stmt::WhileStmtClass:
    case Stmt::DoStmtClass:
    case Stmt::CXXCatchStmtClass:
    case Stmt::GotoStmtClass:
    case Stmt::IndirectGotoStmtClass:
      Reasons |= CognitiveComplexity::Criteria::Increment;
      break;
    default:
      // break LABEL, continue LABEL increase cognitive complexity,
      // but they are not supported in C++ or C.
      // Regular break/continue do not increase cognitive complexity.
      break;
    }

    // B2. Nesting level
    // The following structures increment the nesting level:
    switch (Node->getStmtClass()) {
    // if, else if, else are handled in TraverseIfStmt(),
    // Nested methods and such are handled in TraverseDecl.
    case Stmt::ConditionalOperatorClass:
    case Stmt::SwitchStmtClass:
    case Stmt::ForStmtClass:
    case Stmt::CXXForRangeStmtClass:
    case Stmt::WhileStmtClass:
    case Stmt::DoStmtClass:
    case Stmt::CXXCatchStmtClass:
    case Stmt::LambdaExprClass:
    case Stmt::StmtExprClass:
      Reasons |= CognitiveComplexity::Criteria::IncrementNesting;
      break;
    default:
      break;
    }

    // B3. Nesting increments
    // The following structures receive a nesting increment
    // commensurate with their nested depth inside B2 structures:
    switch (Node->getStmtClass()) {
    // if, else if, else are handled in TraverseIfStmt().
    case Stmt::ConditionalOperatorClass:
    case Stmt::SwitchStmtClass:
    case Stmt::ForStmtClass:
    case Stmt::CXXForRangeStmtClass:
    case Stmt::WhileStmtClass:
    case Stmt::DoStmtClass:
    case Stmt::CXXCatchStmtClass:
      Reasons |= CognitiveComplexity::Criteria::PenalizeNesting;
      break;
    default:
      break;
    }

    if (Node->getStmtClass() == Stmt::ConditionalOperatorClass) {
      // A little beautification.
      // For conditional operator "cond ? true : false" point at the "?"
      // symbol.
      Location = cast<ConditionalOperator>(Node)->getQuestionLoc();
    }

    // If we have found any reasons, let's account it.
    if (Reasons & CognitiveComplexity::Criteria::All)
      CC.account(Location, CurrentNestingLevel, Reasons);

    // Did we decide that the nesting level should be increased?
    if (!(Reasons & CognitiveComplexity::Criteria::IncrementNesting))
      return Base::TraverseStmt(Node);

    return traverseStmtWithIncreasedNestingLevel(Node);
  }

  // The parameter MainAnalyzedFunction is needed to differentiate between the
  // cases where TraverseDecl() is the entry point from
  // FunctionCognitiveComplexityCheck::check() and the cases where it was called
  // from the FunctionASTVisitor itself. Explanation: if we get a function
  // definition (e.g. constructor, destructor, method), the Cognitive Complexity
  // specification states that the Nesting level shall be increased. But if this
  // function is the entry point, then the Nesting level should not be
  // increased. Thus that parameter is there and is used to fall-through
  // directly to traversing if this is the main function that is being analyzed.
  bool TraverseDecl(Decl *Node, bool MainAnalyzedFunction = false) {
    if (!Node || MainAnalyzedFunction)
      return Base::TraverseDecl(Node);

    // B2. Nesting level
    // The following structures increment the nesting level:
    switch (Node->getKind()) {
    case Decl::Function:
    case Decl::CXXMethod:
    case Decl::CXXConstructor:
    case Decl::CXXDestructor:
    case Decl::Block:
      break;
    default:
      // If this is something else, we use early return!
      return Base::TraverseDecl(Node);
      break;
    }

    CC.account(Node->getBeginLoc(), CurrentNestingLevel,
               CognitiveComplexity::Criteria::IncrementNesting);

    return traverseDeclWithIncreasedNestingLevel(Node);
  }

  CognitiveComplexity CC;
};

} // namespace

FunctionCognitiveComplexityCheck::FunctionCognitiveComplexityCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Threshold(Options.get("Threshold", CognitiveComplexity::DefaultLimit)),
      DescribeBasicIncrements(Options.get("DescribeBasicIncrements", true)),
      IgnoreMacros(Options.get("IgnoreMacros", false)) {}

void FunctionCognitiveComplexityCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "Threshold", Threshold);
  Options.store(Opts, "DescribeBasicIncrements", DescribeBasicIncrements);
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void FunctionCognitiveComplexityCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionDecl(isDefinition(),
                   unless(anyOf(isDefaulted(), isDeleted(), isWeak())))
          .bind("func"),
      this);
  Finder->addMatcher(lambdaExpr().bind("lambda"), this);
}

void FunctionCognitiveComplexityCheck::check(
    const MatchFinder::MatchResult &Result) {

  FunctionASTVisitor Visitor(IgnoreMacros);
  SourceLocation Loc;

  const auto *TheDecl = Result.Nodes.getNodeAs<FunctionDecl>("func");
  const auto *TheLambdaExpr = Result.Nodes.getNodeAs<LambdaExpr>("lambda");
  if (TheDecl) {
    assert(TheDecl->hasBody() &&
           "The matchers should only match the functions that "
           "have user-provided body.");
    Loc = TheDecl->getLocation();
    Visitor.TraverseDecl(const_cast<FunctionDecl *>(TheDecl), true);
  } else {
    Loc = TheLambdaExpr->getBeginLoc();
    Visitor.TraverseLambdaExpr(const_cast<LambdaExpr *>(TheLambdaExpr));
  }

  if (Visitor.CC.Total <= Threshold)
    return;

  if (TheDecl)
    diag(Loc, "function %0 has cognitive complexity of %1 (threshold %2)")
        << TheDecl << Visitor.CC.Total << Threshold;
  else
    diag(Loc, "lambda has cognitive complexity of %0 (threshold %1)")
        << Visitor.CC.Total << Threshold;

  if (!DescribeBasicIncrements)
    return;

  // Output all the basic increments of complexity.
  for (const auto &Detail : Visitor.CC.Details) {
    unsigned MsgId = 0;          // The id of the message to output.
    unsigned short Increase = 0; // How much of an increment?
    std::tie(MsgId, Increase) = Detail.process();
    assert(MsgId < Msgs.size() && "MsgId should always be valid");
    // Increase, on the other hand, can be 0.

    diag(Detail.Loc, Msgs[MsgId], DiagnosticIDs::Note)
        << (unsigned)Increase << (unsigned)Detail.Nesting << 1 + Detail.Nesting;
  }
}

} // namespace clang::tidy::readability
