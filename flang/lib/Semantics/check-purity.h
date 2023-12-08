//===-- lib/Semantics/check-purity.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_CHECK_PURITY_H_
#define FORTRAN_SEMANTICS_CHECK_PURITY_H_
#include "flang/Semantics/semantics.h"
#include <list>
namespace Fortran::parser {
struct ExecutableConstruct;
struct SubroutineSubprogram;
struct FunctionSubprogram;
struct PrefixSpec;
} // namespace Fortran::parser
namespace Fortran::semantics {
class PurityChecker : public virtual BaseChecker {
public:
  explicit PurityChecker(SemanticsContext &c) : context_{c} {}
  void Enter(const parser::ExecutableConstruct &);
  void Enter(const parser::SubroutineSubprogram &);
  void Leave(const parser::SubroutineSubprogram &);
  void Enter(const parser::FunctionSubprogram &);
  void Leave(const parser::FunctionSubprogram &);

private:
  bool InPureSubprogram() const;
  bool HasPurePrefix(const std::list<parser::PrefixSpec> &) const;
  void Entered(parser::CharBlock, const std::list<parser::PrefixSpec> &);
  void Left();
  SemanticsContext &context_;
  int depth_{0};
  int pureDepth_{-1};
};
} // namespace Fortran::semantics
#endif
