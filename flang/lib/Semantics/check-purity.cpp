//===-- lib/Semantics/check-purity.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-purity.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"

namespace Fortran::semantics {
void PurityChecker::Enter(const parser::ExecutableConstruct &exec) {
  if (InPureSubprogram() && IsImageControlStmt(exec)) {
    context_.Say(GetImageControlStmtLocation(exec),
        "An image control statement may not appear in a pure subprogram"_err_en_US);
  }
}
void PurityChecker::Enter(const parser::SubroutineSubprogram &subr) {
  const auto &stmt{std::get<parser::Statement<parser::SubroutineStmt>>(subr.t)};
  Entered(
      stmt.source, std::get<std::list<parser::PrefixSpec>>(stmt.statement.t));
}

void PurityChecker::Leave(const parser::SubroutineSubprogram &) { Left(); }

void PurityChecker::Enter(const parser::FunctionSubprogram &func) {
  const auto &stmt{std::get<parser::Statement<parser::FunctionStmt>>(func.t)};
  Entered(
      stmt.source, std::get<std::list<parser::PrefixSpec>>(stmt.statement.t));
}

void PurityChecker::Leave(const parser::FunctionSubprogram &) { Left(); }

bool PurityChecker::InPureSubprogram() const {
  return pureDepth_ >= 0 && depth_ >= pureDepth_;
}

bool PurityChecker::HasPurePrefix(
    const std::list<parser::PrefixSpec> &prefixes) const {
  for (const parser::PrefixSpec &prefix : prefixes) {
    if (std::holds_alternative<parser::PrefixSpec::Pure>(prefix.u)) {
      return true;
    }
  }
  return false;
}

void PurityChecker::Entered(
    parser::CharBlock source, const std::list<parser::PrefixSpec> &prefixes) {
  if (depth_ == 2) {
    context_.messages().Say(source,
        "An internal subprogram may not contain an internal subprogram"_err_en_US);
  }
  if (HasPurePrefix(prefixes)) {
    if (pureDepth_ < 0) {
      pureDepth_ = depth_;
    }
  } else if (InPureSubprogram()) {
    context_.messages().Say(source,
        "An internal subprogram of a pure subprogram must also be pure"_err_en_US);
  }
  ++depth_;
}

void PurityChecker::Left() {
  if (pureDepth_ == --depth_) {
    pureDepth_ = -1;
  }
}

} // namespace Fortran::semantics
