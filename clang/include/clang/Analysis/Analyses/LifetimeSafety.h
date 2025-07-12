//===- LifetimeSafety.h - C++ Lifetime Safety Analysis -*----------- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the entry point for a dataflow-based static analysis
// that checks for C++ lifetime violations.
//
// The analysis is based on the concepts of "origins" and "loans" to track
// pointer lifetimes and detect issues like use-after-free and dangling
// pointers. See the RFC for more details:
// https://discourse.llvm.org/t/rfc-intra-procedural-lifetime-analysis-in-clang/86291
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_H
#include "clang/AST/DeclBase.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
namespace clang {

void runLifetimeSafetyAnalysis(const DeclContext &DC, const CFG &Cfg,
                               AnalysisDeclContext &AC);

} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_H
