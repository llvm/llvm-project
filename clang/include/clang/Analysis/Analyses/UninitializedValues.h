//= UninitializedValues.h - Finding uses of uninitialized values -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines APIs for invoking and reported uninitialized values
// warnings.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNINIT_VALS_H
#define LLVM_CLANG_UNINIT_VALS_H

#include "llvm/ADT/SmallVector.h"

namespace clang {

class AnalysisDeclContext;
class CFG;
class DeclContext;
class Expr;
class VarDecl;

/// A use of a variable, which might be uninitialized.
class UninitUse {
public:
  struct Branch {
    const Stmt *Terminator;
    unsigned Output;
  };

private:
  /// The expression which uses this variable.
  const Expr *User;

  /// Does this use always see an uninitialized value?
  bool AlwaysUninit;

  /// This use is always uninitialized if it occurs after any of these branches
  /// is taken.
  llvm::SmallVector<Branch, 2> UninitBranches;

public:
  UninitUse(const Expr *User, bool AlwaysUninit) :
    User(User), AlwaysUninit(AlwaysUninit) {}

  void addUninitBranch(Branch B) {
    UninitBranches.push_back(B);
  }

  /// Get the expression containing the uninitialized use.
  const Expr *getUser() const { return User; }

  /// The kind of uninitialized use.
  enum Kind {
    /// The use might be uninitialized.
    Maybe,
    /// The use is uninitialized whenever a certain branch is taken.
    Sometimes,
    /// The use is always uninitialized.
    Always
  };

  /// Get the kind of uninitialized use.
  Kind getKind() const {
    return AlwaysUninit ? Always :
           !branch_empty() ? Sometimes : Maybe;
  }

  typedef llvm::SmallVectorImpl<Branch>::const_iterator branch_iterator;
  /// Branches which inevitably result in the variable being used uninitialized.
  branch_iterator branch_begin() const { return UninitBranches.begin(); }
  branch_iterator branch_end() const { return UninitBranches.end(); }
  bool branch_empty() const { return UninitBranches.empty(); }
};

class UninitVariablesHandler {
public:
  UninitVariablesHandler() {}
  virtual ~UninitVariablesHandler();

  /// Called when the uninitialized variable is used at the given expression.
  virtual void handleUseOfUninitVariable(const VarDecl *vd,
                                         const UninitUse &use) {}

  /// Called when the uninitialized variable analysis detects the
  /// idiom 'int x = x'.  All other uses of 'x' within the initializer
  /// are handled by handleUseOfUninitVariable.
  virtual void handleSelfInit(const VarDecl *vd) {}
};

struct UninitVariablesAnalysisStats {
  unsigned NumVariablesAnalyzed;
  unsigned NumBlockVisits;
};

void runUninitializedVariablesAnalysis(const DeclContext &dc, const CFG &cfg,
                                       AnalysisDeclContext &ac,
                                       UninitVariablesHandler &handler,
                                       UninitVariablesAnalysisStats &stats);

}
#endif
