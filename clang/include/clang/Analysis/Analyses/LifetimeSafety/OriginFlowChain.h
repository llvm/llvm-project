//====- OriginFlowChain.h - C++ Lifetime Safety Checker --------*- C++ -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines buildOriginFlowChain, which is used to build the
// propagation flow of a given Loan within a specified Origin, starting
// from a particular ProgramPoint.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_ORIGINFLOWCHAIN_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_ORIGINFLOWCHAIN_H

#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LoanPropagation.h"

namespace clang::lifetimes::internal {

/// Builds a chain of origin flows for a specific loan. It tracks how the Loan
/// moves and transforms.
///
/// This function starts from a given ProgramPoint and builds the propagation
/// flow of the specified LoanID within the context of a given OriginID.
llvm::SmallVector<OriginID> buildOriginFlowChain(
    const FactManager &FactMgr, const LoanPropagationAnalysis &LoanPropagation,
    ProgramPoint StartPoint, const OriginID StartOID, const LoanID TargetLoan);
} // namespace clang::lifetimes::internal

#endif
