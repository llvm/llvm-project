//===--- SemaStmtAttr.cpp - Statement Attribute Handling ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements stmt-related attribute processing.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/DelayedDiagnostic.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/ScopeInfo.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang;
using namespace sema;

static Attr *handleFallThroughAttr(Sema &S, Stmt *St, const ParsedAttr &A,
                                   SourceRange Range) {
  FallThroughAttr Attr(S.Context, A);
  if (!isa<NullStmt>(St)) {
    S.Diag(A.getRange().getBegin(), diag::err_fallthrough_attr_wrong_target)
        << Attr.getSpelling() << St->getBeginLoc();
    if (isa<SwitchCase>(St)) {
      SourceLocation L = S.getLocForEndOfToken(Range.getEnd());
      S.Diag(L, diag::note_fallthrough_insert_semi_fixit)
          << FixItHint::CreateInsertion(L, ";");
    }
    return nullptr;
  }
  auto *FnScope = S.getCurFunction();
  if (FnScope->SwitchStack.empty()) {
    S.Diag(A.getRange().getBegin(), diag::err_fallthrough_attr_outside_switch);
    return nullptr;
  }

  // If this is spelled as the standard C++17 attribute, but not in C++17, warn
  // about using it as an extension.
  if (!S.getLangOpts().CPlusPlus17 && A.isCXX11Attribute() &&
      !A.getScopeName())
    S.Diag(A.getLoc(), diag::ext_cxx17_attr) << A;

  FnScope->setHasFallthroughStmt();
  return ::new (S.Context) FallThroughAttr(S.Context, A);
}

static Attr *handleSuppressAttr(Sema &S, Stmt *St, const ParsedAttr &A,
                                SourceRange Range) {
  if (A.getNumArgs() < 1) {
    S.Diag(A.getLoc(), diag::err_attribute_too_few_arguments) << A << 1;
    return nullptr;
  }

  std::vector<StringRef> DiagnosticIdentifiers;
  for (unsigned I = 0, E = A.getNumArgs(); I != E; ++I) {
    StringRef RuleName;

    if (!S.checkStringLiteralArgumentAttr(A, I, RuleName, nullptr))
      return nullptr;

    // FIXME: Warn if the rule name is unknown. This is tricky because only
    // clang-tidy knows about available rules.
    DiagnosticIdentifiers.push_back(RuleName);
  }

  return ::new (S.Context) SuppressAttr(
      S.Context, A, DiagnosticIdentifiers.data(), DiagnosticIdentifiers.size());
}

static Attr *handleLoopHintAttr(Sema &S, Stmt *St, const ParsedAttr &A,
                                SourceRange) {
  IdentifierLoc *PragmaNameLoc = A.getArgAsIdent(0);
  IdentifierLoc *OptionLoc = A.getArgAsIdent(1);
  IdentifierLoc *StateLoc = A.getArgAsIdent(2);
  Expr *ValueExpr = A.getArgAsExpr(3);

  StringRef PragmaName =
      llvm::StringSwitch<StringRef>(PragmaNameLoc->Ident->getName())
          .Cases("unroll", "nounroll", "unroll_and_jam", "nounroll_and_jam",
                 PragmaNameLoc->Ident->getName())
          .Default("clang loop");

  if (St->getStmtClass() != Stmt::DoStmtClass &&
      St->getStmtClass() != Stmt::ForStmtClass &&
      St->getStmtClass() != Stmt::CXXForRangeStmtClass &&
      St->getStmtClass() != Stmt::WhileStmtClass) {
    std::string Pragma = "#pragma " + std::string(PragmaName);
    S.Diag(St->getBeginLoc(), diag::err_pragma_loop_precedes_nonloop) << Pragma;
    return nullptr;
  }

  LoopHintAttr::OptionType Option;
  LoopHintAttr::LoopHintState State;

  auto SetHints = [&Option, &State](LoopHintAttr::OptionType O,
                                    LoopHintAttr::LoopHintState S) {
    Option = O;
    State = S;
  };

  if (PragmaName == "nounroll") {
    SetHints(LoopHintAttr::Unroll, LoopHintAttr::Disable);
  } else if (PragmaName == "unroll") {
    // #pragma unroll N
    if (ValueExpr)
      SetHints(LoopHintAttr::UnrollCount, LoopHintAttr::Numeric);
    else
      SetHints(LoopHintAttr::Unroll, LoopHintAttr::Enable);
  } else if (PragmaName == "nounroll_and_jam") {
    SetHints(LoopHintAttr::UnrollAndJam, LoopHintAttr::Disable);
  } else if (PragmaName == "unroll_and_jam") {
    // #pragma unroll_and_jam N
    if (ValueExpr)
      SetHints(LoopHintAttr::UnrollAndJamCount, LoopHintAttr::Numeric);
    else
      SetHints(LoopHintAttr::UnrollAndJam, LoopHintAttr::Enable);
  } else {
    // #pragma clang loop ...
    assert(OptionLoc && OptionLoc->Ident &&
           "Attribute must have valid option info.");
    Option = llvm::StringSwitch<LoopHintAttr::OptionType>(
                 OptionLoc->Ident->getName())
                 .Case("vectorize", LoopHintAttr::Vectorize)
                 .Case("vectorize_width", LoopHintAttr::VectorizeWidth)
                 .Case("interleave", LoopHintAttr::Interleave)
                 .Case("vectorize_predicate", LoopHintAttr::VectorizePredicate)
                 .Case("interleave_count", LoopHintAttr::InterleaveCount)
                 .Case("unroll", LoopHintAttr::Unroll)
                 .Case("unroll_count", LoopHintAttr::UnrollCount)
                 .Case("pipeline", LoopHintAttr::PipelineDisabled)
                 .Case("pipeline_initiation_interval",
                       LoopHintAttr::PipelineInitiationInterval)
                 .Case("distribute", LoopHintAttr::Distribute)
                 .Default(LoopHintAttr::Vectorize);
    if (Option == LoopHintAttr::VectorizeWidth ||
        Option == LoopHintAttr::InterleaveCount ||
        Option == LoopHintAttr::UnrollCount ||
        Option == LoopHintAttr::PipelineInitiationInterval) {
      assert(ValueExpr && "Attribute must have a valid value expression.");
      if (S.CheckLoopHintExpr(ValueExpr, St->getBeginLoc()))
        return nullptr;
      State = LoopHintAttr::Numeric;
    } else if (Option == LoopHintAttr::Vectorize ||
               Option == LoopHintAttr::Interleave ||
               Option == LoopHintAttr::VectorizePredicate ||
               Option == LoopHintAttr::Unroll ||
               Option == LoopHintAttr::Distribute ||
               Option == LoopHintAttr::PipelineDisabled) {
      assert(StateLoc && StateLoc->Ident && "Loop hint must have an argument");
      if (StateLoc->Ident->isStr("disable"))
        State = LoopHintAttr::Disable;
      else if (StateLoc->Ident->isStr("assume_safety"))
        State = LoopHintAttr::AssumeSafety;
      else if (StateLoc->Ident->isStr("full"))
        State = LoopHintAttr::Full;
      else if (StateLoc->Ident->isStr("enable"))
        State = LoopHintAttr::Enable;
      else
        llvm_unreachable("bad loop hint argument");
    } else
      llvm_unreachable("bad loop hint");
  }

  return LoopHintAttr::CreateImplicit(S.Context, Option, State, ValueExpr, A);
}

namespace {
class CallExprFinder : public ConstEvaluatedExprVisitor<CallExprFinder> {
  bool FoundCallExpr = false;

public:
  typedef ConstEvaluatedExprVisitor<CallExprFinder> Inherited;

  CallExprFinder(Sema &S, const Stmt *St) : Inherited(S.Context) { Visit(St); }

  bool foundCallExpr() { return FoundCallExpr; }

  void VisitCallExpr(const CallExpr *E) { FoundCallExpr = true; }
  void VisitAsmStmt(const AsmStmt *S) { FoundCallExpr = true; }

  void Visit(const Stmt *St) {
    if (!St)
      return;
    ConstEvaluatedExprVisitor<CallExprFinder>::Visit(St);
  }
};
} // namespace

static Attr *handleNoMergeAttr(Sema &S, Stmt *St, const ParsedAttr &A,
                               SourceRange Range) {
  NoMergeAttr NMA(S.Context, A);
  if (S.CheckAttrNoArgs(A))
    return nullptr;

  CallExprFinder CEF(S, St);

  if (!CEF.foundCallExpr()) {
    S.Diag(St->getBeginLoc(), diag::warn_nomerge_attribute_ignored_in_stmt)
        << NMA.getSpelling();
    return nullptr;
  }

  return ::new (S.Context) NoMergeAttr(S.Context, A);
}

static Attr *handleLikely(Sema &S, Stmt *St, const ParsedAttr &A,
                          SourceRange Range) {

  if (!S.getLangOpts().CPlusPlus20 && A.isCXX11Attribute() && !A.getScopeName())
    S.Diag(A.getLoc(), diag::ext_cxx20_attr) << A << Range;

  return ::new (S.Context) LikelyAttr(S.Context, A);
}

static Attr *handleUnlikely(Sema &S, Stmt *St, const ParsedAttr &A,
                            SourceRange Range) {

  if (!S.getLangOpts().CPlusPlus20 && A.isCXX11Attribute() && !A.getScopeName())
    S.Diag(A.getLoc(), diag::ext_cxx20_attr) << A << Range;

  return ::new (S.Context) UnlikelyAttr(S.Context, A);
}

static void
CheckForIncompatibleAttributes(Sema &S,
                               const SmallVectorImpl<const Attr *> &Attrs) {
  // There are 6 categories of loop hints attributes: vectorize, interleave,
  // unroll, unroll_and_jam, pipeline and distribute. Except for distribute they
  // come in two variants: a state form and a numeric form.  The state form
  // selectively defaults/enables/disables the transformation for the loop
  // (for unroll, default indicates full unrolling rather than enabling the
  // transformation). The numeric form form provides an integer hint (for
  // example, unroll count) to the transformer. The following array accumulates
  // the hints encountered while iterating through the attributes to check for
  // compatibility.
  struct {
    const LoopHintAttr *StateAttr;
    const LoopHintAttr *NumericAttr;
  } HintAttrs[] = {{nullptr, nullptr}, {nullptr, nullptr}, {nullptr, nullptr},
                   {nullptr, nullptr}, {nullptr, nullptr}, {nullptr, nullptr},
                   {nullptr, nullptr}};

  for (const auto *I : Attrs) {
    const LoopHintAttr *LH = dyn_cast<LoopHintAttr>(I);

    // Skip non loop hint attributes
    if (!LH)
      continue;

    LoopHintAttr::OptionType Option = LH->getOption();
    enum {
      Vectorize,
      Interleave,
      Unroll,
      UnrollAndJam,
      Distribute,
      Pipeline,
      VectorizePredicate
    } Category;
    switch (Option) {
    case LoopHintAttr::Vectorize:
    case LoopHintAttr::VectorizeWidth:
      Category = Vectorize;
      break;
    case LoopHintAttr::Interleave:
    case LoopHintAttr::InterleaveCount:
      Category = Interleave;
      break;
    case LoopHintAttr::Unroll:
    case LoopHintAttr::UnrollCount:
      Category = Unroll;
      break;
    case LoopHintAttr::UnrollAndJam:
    case LoopHintAttr::UnrollAndJamCount:
      Category = UnrollAndJam;
      break;
    case LoopHintAttr::Distribute:
      // Perform the check for duplicated 'distribute' hints.
      Category = Distribute;
      break;
    case LoopHintAttr::PipelineDisabled:
    case LoopHintAttr::PipelineInitiationInterval:
      Category = Pipeline;
      break;
    case LoopHintAttr::VectorizePredicate:
      Category = VectorizePredicate;
      break;
    };

    assert(Category < sizeof(HintAttrs) / sizeof(HintAttrs[0]));
    auto &CategoryState = HintAttrs[Category];
    const LoopHintAttr *PrevAttr;
    if (Option == LoopHintAttr::Vectorize ||
        Option == LoopHintAttr::Interleave || Option == LoopHintAttr::Unroll ||
        Option == LoopHintAttr::UnrollAndJam ||
        Option == LoopHintAttr::VectorizePredicate ||
        Option == LoopHintAttr::PipelineDisabled ||
        Option == LoopHintAttr::Distribute) {
      // Enable|Disable|AssumeSafety hint.  For example, vectorize(enable).
      PrevAttr = CategoryState.StateAttr;
      CategoryState.StateAttr = LH;
    } else {
      // Numeric hint.  For example, vectorize_width(8).
      PrevAttr = CategoryState.NumericAttr;
      CategoryState.NumericAttr = LH;
    }

    PrintingPolicy Policy(S.Context.getLangOpts());
    SourceLocation OptionLoc = LH->getRange().getBegin();
    if (PrevAttr)
      // Cannot specify same type of attribute twice.
      S.Diag(OptionLoc, diag::err_pragma_loop_compatibility)
          << /*Duplicate=*/true << PrevAttr->getDiagnosticName(Policy)
          << LH->getDiagnosticName(Policy);

    if (CategoryState.StateAttr && CategoryState.NumericAttr &&
        (Category == Unroll || Category == UnrollAndJam ||
         CategoryState.StateAttr->getState() == LoopHintAttr::Disable)) {
      // Disable hints are not compatible with numeric hints of the same
      // category.  As a special case, numeric unroll hints are also not
      // compatible with enable or full form of the unroll pragma because these
      // directives indicate full unrolling.
      S.Diag(OptionLoc, diag::err_pragma_loop_compatibility)
          << /*Duplicate=*/false
          << CategoryState.StateAttr->getDiagnosticName(Policy)
          << CategoryState.NumericAttr->getDiagnosticName(Policy);
    }
  }

  // C++20 [dcl.attr.likelihood]p1 The attribute-token likely shall not appear
  // in an attribute-specifier-seq that contains the attribute-token unlikely.
  const LikelyAttr *Likely = nullptr;
  const UnlikelyAttr *Unlikely = nullptr;
  for (const auto *I : Attrs) {
    if (const auto *Attr = dyn_cast<LikelyAttr>(I)) {
      if (Unlikely) {
        S.Diag(Attr->getLocation(), diag::err_attributes_are_not_compatible)
            << Attr << Unlikely << Attr->getRange();
        S.Diag(Unlikely->getLocation(), diag::note_conflicting_attribute)
            << Unlikely->getRange();
        return;
      }
      Likely = Attr;
    } else if (const auto *Attr = dyn_cast<UnlikelyAttr>(I)) {
      if (Likely) {
        S.Diag(Attr->getLocation(), diag::err_attributes_are_not_compatible)
            << Attr << Likely << Attr->getRange();
        S.Diag(Likely->getLocation(), diag::note_conflicting_attribute)
            << Likely->getRange();
        return;
      }
      Unlikely = Attr;
    }
  }
}

static Attr *handleOpenCLUnrollHint(Sema &S, Stmt *St, const ParsedAttr &A,
                                    SourceRange Range) {
  // Although the feature was introduced only in OpenCL C v2.0 s6.11.5, it's
  // useful for OpenCL 1.x too and doesn't require HW support.
  // opencl_unroll_hint can have 0 arguments (compiler
  // determines unrolling factor) or 1 argument (the unroll factor provided
  // by the user).

  unsigned NumArgs = A.getNumArgs();

  if (NumArgs > 1) {
    S.Diag(A.getLoc(), diag::err_attribute_too_many_arguments) << A << 1;
    return nullptr;
  }

  unsigned UnrollFactor = 0;

  if (NumArgs == 1) {
    Expr *E = A.getArgAsExpr(0);
    Optional<llvm::APSInt> ArgVal;

    if (!(ArgVal = E->getIntegerConstantExpr(S.Context))) {
      S.Diag(A.getLoc(), diag::err_attribute_argument_type)
          << A << AANT_ArgumentIntegerConstant << E->getSourceRange();
      return nullptr;
    }

    int Val = ArgVal->getSExtValue();

    if (Val <= 0) {
      S.Diag(A.getRange().getBegin(),
             diag::err_attribute_requires_positive_integer)
          << A << /* positive */ 0;
      return nullptr;
    }
    UnrollFactor = Val;
  }

  return OpenCLUnrollHintAttr::CreateImplicit(S.Context, UnrollFactor);
}

static Attr *ProcessStmtAttribute(Sema &S, Stmt *St, const ParsedAttr &A,
                                  SourceRange Range) {
  switch (A.getKind()) {
  case ParsedAttr::UnknownAttribute:
    S.Diag(A.getLoc(), A.isDeclspecAttribute()
                           ? (unsigned)diag::warn_unhandled_ms_attribute_ignored
                           : (unsigned)diag::warn_unknown_attribute_ignored)
        << A << A.getRange();
    return nullptr;
  case ParsedAttr::AT_FallThrough:
    return handleFallThroughAttr(S, St, A, Range);
  case ParsedAttr::AT_LoopHint:
    return handleLoopHintAttr(S, St, A, Range);
  case ParsedAttr::AT_OpenCLUnrollHint:
    return handleOpenCLUnrollHint(S, St, A, Range);
  case ParsedAttr::AT_Suppress:
    return handleSuppressAttr(S, St, A, Range);
  case ParsedAttr::AT_NoMerge:
    return handleNoMergeAttr(S, St, A, Range);
  case ParsedAttr::AT_Likely:
    return handleLikely(S, St, A, Range);
  case ParsedAttr::AT_Unlikely:
    return handleUnlikely(S, St, A, Range);
  default:
    // if we're here, then we parsed a known attribute, but didn't recognize
    // it as a statement attribute => it is declaration attribute
    S.Diag(A.getRange().getBegin(), diag::err_decl_attribute_invalid_on_stmt)
        << A << St->getBeginLoc();
    return nullptr;
  }
}

StmtResult Sema::ProcessStmtAttributes(Stmt *S,
                                       const ParsedAttributesView &AttrList,
                                       SourceRange Range) {
  SmallVector<const Attr*, 8> Attrs;
  for (const ParsedAttr &AL : AttrList) {
    if (Attr *a = ProcessStmtAttribute(*this, S, AL, Range))
      Attrs.push_back(a);
  }

  CheckForIncompatibleAttributes(*this, Attrs);

  if (Attrs.empty())
    return S;

  return ActOnAttributedStmt(Range.getBegin(), Attrs, S);
}
