//===--- AvoidCommonBlocks.cpp - flang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidCommonBlocks.h"

namespace Fortran::tidy::modernize {

using namespace parser::literals;
void AvoidCommonBlocksCheck::Enter(const parser::CommonStmt &) {
  if (context()->getSemanticsContext().location().has_value()) {
    Say(context()->getSemanticsContext().location().value(),
        "Common blocks are not recommended"_warn_en_US);
  }
}

} // namespace Fortran::tidy::modernize
