//===-- lib/Parser/debug-parser.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "debug-parser.h"
#include "flang/Parser/user-state.h"
#include <string>

namespace Fortran::parser {

std::optional<Success> DebugParser::Parse(ParseState &state) const {
  if (auto ustate{state.userState()}) {
    if (auto out{ustate->debugOutput()}) {
      std::string note{str_, length_};
      Message message{state.GetLocation(), "parser debug: %s"_en_US, note};
      message.SetContext(state.context().get());
      message.Emit(*out, ustate->allCooked(), true);
    }
  }
  return Success{};
}
} // namespace Fortran::parser
