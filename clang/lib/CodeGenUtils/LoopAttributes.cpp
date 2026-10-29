//===--- LoopAttributes.cpp - Shared loop hint state ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenUtils/LoopAttributes.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/CodeGenOptions.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang::CodeGenUtils {

LoopAttributes::LoopAttributes(bool IsParallel)
    : IsParallel(IsParallel), VectorizeEnable(LoopAttributes::Unspecified),
      UnrollEnable(LoopAttributes::Unspecified),
      UnrollAndJamEnable(LoopAttributes::Unspecified),
      VectorizePredicateEnable(LoopAttributes::Unspecified), VectorizeWidth(0),
      VectorizeScalable(LoopAttributes::Unspecified), InterleaveCount(0),
      UnrollCount(0), UnrollAndJamCount(0),
      DistributeEnable(LoopAttributes::Unspecified), PipelineDisabled(false),
      LICMDisabled(false), PipelineInitiationInterval(0), CodeAlign(0),
      MustProgress(false) {}

void LoopAttributes::clear() {
  IsParallel = false;
  VectorizeWidth = 0;
  VectorizeScalable = LoopAttributes::Unspecified;
  InterleaveCount = 0;
  UnrollCount = 0;
  UnrollAndJamCount = 0;
  VectorizeEnable = LoopAttributes::Unspecified;
  UnrollEnable = LoopAttributes::Unspecified;
  UnrollAndJamEnable = LoopAttributes::Unspecified;
  VectorizePredicateEnable = LoopAttributes::Unspecified;
  DistributeEnable = LoopAttributes::Unspecified;
  PipelineDisabled = false;
  LICMDisabled = false;
  PipelineInitiationInterval = 0;
  CodeAlign = 0;
  MustProgress = false;
}

void updateLoopAttributes(LoopAttributes &Attrs, ASTContext &Ctx,
                          const CodeGenOptions &CGOpts,
                          llvm::ArrayRef<const Attr *> StmtAttrs,
                          bool MustProgress) {
  // Identify loop hint attributes from Attrs.
  for (const auto *Attr : StmtAttrs) {
    const LoopHintAttr *LH = dyn_cast<LoopHintAttr>(Attr);
    const OpenCLUnrollHintAttr *OpenCLHint =
        dyn_cast<OpenCLUnrollHintAttr>(Attr);
    const HLSLLoopHintAttr *HLSLLoopHint = dyn_cast<HLSLLoopHintAttr>(Attr);
    // Skip non loop hint attributes
    if (!LH && !OpenCLHint && !HLSLLoopHint) {
      continue;
    }

    LoopHintAttr::OptionType Option = LoopHintAttr::Unroll;
    LoopHintAttr::LoopHintState State = LoopHintAttr::Disable;
    unsigned ValueInt = 1;
    // Translate opencl_unroll_hint attribute argument to
    // equivalent LoopHintAttr enums.
    // OpenCL v2.0 s6.11.5:
    // 0 - enable unroll (no argument).
    // 1 - disable unroll.
    // other positive integer n - unroll by n.
    if (OpenCLHint) {
      ValueInt = OpenCLHint->getUnrollHint();
      if (ValueInt == 0) {
        State = LoopHintAttr::Enable;
      } else if (ValueInt != 1) {
        Option = LoopHintAttr::UnrollCount;
        State = LoopHintAttr::Numeric;
      }
    } else if (HLSLLoopHint) {
      ValueInt = HLSLLoopHint->getDirective();
      if (HLSLLoopHint->getSemanticSpelling() ==
          HLSLLoopHintAttr::Spelling::Microsoft_unroll) {
        if (ValueInt == 0)
          State = LoopHintAttr::Enable;
        if (ValueInt > 0) {
          Option = LoopHintAttr::UnrollCount;
          State = LoopHintAttr::Numeric;
        }
      }
    } else if (LH) {
      auto *ValueExpr = LH->getValue();
      if (ValueExpr) {
        llvm::APSInt ValueAPS = ValueExpr->EvaluateKnownConstInt(Ctx);
        ValueInt = ValueAPS.getSExtValue();
      }

      Option = LH->getOption();
      State = LH->getState();
    }
    switch (State) {
    case LoopHintAttr::Disable:
      switch (Option) {
      case LoopHintAttr::Vectorize:
        // Disable vectorization by specifying a width of 1 and disabling
        // scalable vectorization explicitly.
        Attrs.VectorizeWidth = 1;
        Attrs.VectorizeScalable = LoopAttributes::Disable;
        break;
      case LoopHintAttr::Interleave:
        // Disable interleaving by speciyfing a count of 1.
        Attrs.InterleaveCount = 1;
        break;
      case LoopHintAttr::Unroll:
        Attrs.UnrollEnable = LoopAttributes::Disable;
        break;
      case LoopHintAttr::UnrollAndJam:
        Attrs.UnrollAndJamEnable = LoopAttributes::Disable;
        break;
      case LoopHintAttr::VectorizePredicate:
        Attrs.VectorizePredicateEnable = LoopAttributes::Disable;
        break;
      case LoopHintAttr::Distribute:
        Attrs.DistributeEnable = LoopAttributes::Disable;
        break;
      case LoopHintAttr::PipelineDisabled:
        Attrs.PipelineDisabled = true;
        break;
      case LoopHintAttr::LICMDisabled:
        Attrs.LICMDisabled = true;
        break;
      case LoopHintAttr::UnrollCount:
      case LoopHintAttr::UnrollAndJamCount:
      case LoopHintAttr::VectorizeWidth:
      case LoopHintAttr::InterleaveCount:
      case LoopHintAttr::PipelineInitiationInterval:
        llvm_unreachable("Options cannot be disabled.");
        break;
      }
      break;
    case LoopHintAttr::Enable:
      switch (Option) {
      case LoopHintAttr::Vectorize:
      case LoopHintAttr::Interleave:
        Attrs.VectorizeEnable = LoopAttributes::Enable;
        break;
      case LoopHintAttr::Unroll:
        Attrs.UnrollEnable = LoopAttributes::Enable;
        break;
      case LoopHintAttr::UnrollAndJam:
        Attrs.UnrollAndJamEnable = LoopAttributes::Enable;
        break;
      case LoopHintAttr::VectorizePredicate:
        Attrs.VectorizePredicateEnable = LoopAttributes::Enable;
        break;
      case LoopHintAttr::Distribute:
        Attrs.DistributeEnable = LoopAttributes::Enable;
        break;
      case LoopHintAttr::UnrollCount:
      case LoopHintAttr::UnrollAndJamCount:
      case LoopHintAttr::VectorizeWidth:
      case LoopHintAttr::InterleaveCount:
      case LoopHintAttr::PipelineDisabled:
      case LoopHintAttr::PipelineInitiationInterval:
      case LoopHintAttr::LICMDisabled:
        llvm_unreachable("Options cannot enabled.");
        break;
      }
      break;
    case LoopHintAttr::AssumeSafety:
      switch (Option) {
      case LoopHintAttr::Vectorize:
      case LoopHintAttr::Interleave:
        // Apply "llvm.mem.parallel_loop_access" metadata to load/stores.
        Attrs.IsParallel = true;
        Attrs.VectorizeEnable = LoopAttributes::Enable;
        break;
      case LoopHintAttr::Unroll:
      case LoopHintAttr::UnrollAndJam:
      case LoopHintAttr::VectorizePredicate:
      case LoopHintAttr::UnrollCount:
      case LoopHintAttr::UnrollAndJamCount:
      case LoopHintAttr::VectorizeWidth:
      case LoopHintAttr::InterleaveCount:
      case LoopHintAttr::Distribute:
      case LoopHintAttr::PipelineDisabled:
      case LoopHintAttr::PipelineInitiationInterval:
      case LoopHintAttr::LICMDisabled:
        llvm_unreachable("Options cannot be used to assume mem safety.");
        break;
      }
      break;
    case LoopHintAttr::Full:
      switch (Option) {
      case LoopHintAttr::Unroll:
        Attrs.UnrollEnable = LoopAttributes::Full;
        break;
      case LoopHintAttr::UnrollAndJam:
        Attrs.UnrollAndJamEnable = LoopAttributes::Full;
        break;
      case LoopHintAttr::Vectorize:
      case LoopHintAttr::Interleave:
      case LoopHintAttr::UnrollCount:
      case LoopHintAttr::UnrollAndJamCount:
      case LoopHintAttr::VectorizeWidth:
      case LoopHintAttr::InterleaveCount:
      case LoopHintAttr::Distribute:
      case LoopHintAttr::PipelineDisabled:
      case LoopHintAttr::PipelineInitiationInterval:
      case LoopHintAttr::VectorizePredicate:
      case LoopHintAttr::LICMDisabled:
        llvm_unreachable("Options cannot be used with 'full' hint.");
        break;
      }
      break;
    case LoopHintAttr::FixedWidth:
    case LoopHintAttr::ScalableWidth:
      switch (Option) {
      case LoopHintAttr::VectorizeWidth:
        Attrs.VectorizeScalable = State == LoopHintAttr::ScalableWidth
                                      ? LoopAttributes::Enable
                                      : LoopAttributes::Disable;
        if (LH->getValue())
          Attrs.VectorizeWidth = ValueInt;
        break;
      default:
        llvm_unreachable("Options cannot be used with 'scalable' hint.");
        break;
      }
      break;
    case LoopHintAttr::Numeric:
      switch (Option) {
      case LoopHintAttr::InterleaveCount:
        Attrs.InterleaveCount = ValueInt;
        break;
      case LoopHintAttr::UnrollCount:
        Attrs.UnrollCount = ValueInt;
        break;
      case LoopHintAttr::UnrollAndJamCount:
        Attrs.UnrollAndJamCount = ValueInt;
        break;
      case LoopHintAttr::PipelineInitiationInterval:
        Attrs.PipelineInitiationInterval = ValueInt;
        break;
      case LoopHintAttr::Unroll:
      case LoopHintAttr::UnrollAndJam:
      case LoopHintAttr::VectorizePredicate:
      case LoopHintAttr::Vectorize:
      case LoopHintAttr::VectorizeWidth:
      case LoopHintAttr::Interleave:
      case LoopHintAttr::Distribute:
      case LoopHintAttr::PipelineDisabled:
      case LoopHintAttr::LICMDisabled:
        llvm_unreachable("Options cannot be assigned a value.");
        break;
      }
      break;
    }
  }

  // Identify loop attribute 'code_align' from Attrs.
  // For attribute code_align:
  // n - 'llvm.loop.align i32 n' metadata will be emitted.
  // A source-level [[clang::code_align]] attribute takes precedence over the
  // -falign-loops=N command-line default.
  if (const auto *CodeAlign = getSpecificAttr<CodeAlignAttr>(StmtAttrs)) {
    const auto *CE = cast<ConstantExpr>(CodeAlign->getAlignment());
    llvm::APSInt ArgVal = CE->getResultAsAPSInt();
    Attrs.CodeAlign = ArgVal.getSExtValue();
  } else if (CGOpts.LoopAlignment) {
    Attrs.CodeAlign = CGOpts.LoopAlignment;
  }

  Attrs.MustProgress = MustProgress;

  if (CGOpts.OptimizationLevel > 0)
    // Disable unrolling for the loop, if unrolling is disabled (via
    // -fno-unroll-loops) and no pragmas override the decision.
    if (!CGOpts.UnrollLoops &&
        (Attrs.UnrollEnable == LoopAttributes::Unspecified &&
         Attrs.UnrollCount == 0))
      Attrs.UnrollEnable = LoopAttributes::Disable;
}

} // namespace clang::CodeGenUtils
