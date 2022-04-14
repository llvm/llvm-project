//=- CIRBasedWarnings.h - Sema warnings based on libAnalysis -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines CIRBasedWarnings, a worker object used by Sema
// that issues warnings based on dataflow-analysis.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_CIRBASEDWARNINGS_H
#define LLVM_CLANG_SEMA_CIRBASEDWARNINGS_H

#include "clang/Sema/AnalysisBasedWarnings.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>

namespace cir {
class CIRGenerator;
} // namespace cir
namespace clang {

class BlockExpr;
class Decl;
class FunctionDecl;
class ObjCMethodDecl;
class QualType;
class Sema;

namespace sema {

class FunctionScopeInfo;

class CIRBasedWarnings {
private:
  Sema &S;
  AnalysisBasedWarnings::Policy DefaultPolicy;
  // std::unique_ptr<cir::CIRGenerator> CIRGen;

  //class InterProceduralData;
  //std::unique_ptr<InterProceduralData> IPData;

  enum VisitFlag { NotVisited = 0, Visited = 1, Pending = 2 };
  llvm::DenseMap<const FunctionDecl*, VisitFlag> VisitedFD;

  /// @}

public:
  CIRBasedWarnings(Sema &s);
  ~CIRBasedWarnings();

  void IssueWarnings(AnalysisBasedWarnings::Policy P, FunctionScopeInfo *fscope,
                     const Decl *D, QualType BlockType);

  //Policy getDefaultPolicy() { return DefaultPolicy; }

  void PrintStats() const;
};

} // namespace sema
} // namespace clang

#endif
