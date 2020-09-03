//===-- lib/Parser/type-parser-implementation.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Macros for implementing per-type parsers

#ifndef FORTRAN_PARSER_TYPE_PARSER_IMPLEMENTATION_H_
#define FORTRAN_PARSER_TYPE_PARSER_IMPLEMENTATION_H_

#include "type-parsers.h"

#undef TYPE_PARSER
#undef TYPE_CONTEXT_PARSER

// The result type of a parser combinator expression is determined
// here via "decltype(attempt(pexpr))" to work around a g++ bug that
// causes it to crash on "decltype(pexpr)" when pexpr's top-level
// operator is an overridden || of parsing alternatives.
#define TYPE_PARSER(pexpr) \
  template <> \
  auto Parser<typename decltype(attempt(pexpr))::resultType>::Parse( \
      ParseState &state) \
      ->std::optional<resultType> { \
    static constexpr auto parser{(pexpr)}; \
    return parser.Parse(state); \
  }

#define TYPE_CONTEXT_PARSER(contextText, pexpr) \
  TYPE_PARSER(CONTEXT_PARSER((contextText), (pexpr)))

#endif
