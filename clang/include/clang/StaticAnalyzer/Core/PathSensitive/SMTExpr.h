//== SMTExpr.h --------------------------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a SMT generic Expr API, which will be the base class
//  for every SMT solver expr specific class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SMTEXPR_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SMTEXPR_H

#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"

namespace clang {
namespace ento {

class SMTExpr {
public:
  SMTExpr() = default;
  virtual ~SMTExpr() = default;

  bool operator<(const SMTExpr &Other) const {
    llvm::FoldingSetNodeID ID1, ID2;
    Profile(ID1);
    Other.Profile(ID2);
    return ID1 < ID2;
  }

  virtual void Profile(llvm::FoldingSetNodeID &ID) const {
    static int Tag = 0;
    ID.AddPointer(&Tag);
  }

  friend bool operator==(SMTExpr const &LHS, SMTExpr const &RHS) {
    return LHS.equal_to(RHS);
  }

  virtual void print(raw_ostream &OS) const = 0;

  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }

protected:
  virtual bool equal_to(SMTExpr const &other) const = 0;
};

using SMTExprRef = std::shared_ptr<SMTExpr>;

} // namespace ento
} // namespace clang

#endif
