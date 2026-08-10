//===-- lib/Parser/debug-parser.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_DEBUG_PARSER_H_
#define FORTRAN_PARSER_DEBUG_PARSER_H_

// Implements the parser with syntax "(YOUR MESSAGE HERE)"_debug for use
// in temporary modifications to the grammar intended for tracing the
// flow of the parsers.  Not to be used in production.

#include "basic-parsers.h"
#include "flang/Parser/parse-state.h"
#include <cstddef>
#include <optional>

namespace Fortran::parser {

class DebugParser {
public:
  using resultType = Success;
  constexpr DebugParser(const DebugParser &) = default;
  constexpr DebugParser(const char *str, std::size_t n)
      : str_{str}, length_{n} {}
  std::optional<Success> Parse(ParseState &) const;

private:
  const char *const str_;
  const std::size_t length_;
};

constexpr DebugParser operator""_debug(const char str[], std::size_t n) {
  return DebugParser{str, n};
}
} // namespace Fortran::parser
#endif // FORTRAN_PARSER_DEBUG_PARSER_H_
