//===-- include/flang/Parser/options.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_OPTIONS_H_
#define FORTRAN_PARSER_OPTIONS_H_

#include "characters.h"
#include "flang/Support/Fortran-features.h"

#include <optional>
#include <string>
#include <vector>

namespace Fortran::parser {

struct Options {
  Options() {}

  using Predefinition = std::pair<std::string, std::optional<std::string>>;

  bool isFixedForm{false};
  int fixedFormColumns{72};
  common::LanguageFeatureControl features;
  std::vector<std::string> searchDirectories;
  std::vector<std::string> intrinsicModuleDirectories;
  std::vector<Predefinition> predefinitions;
  bool instrumentedParse{false};
  bool isModuleFile{false};
  bool needProvenanceRangeToCharBlockMappings{false};
  Fortran::parser::Encoding encoding{Fortran::parser::Encoding::UTF_8};
  bool prescanAndReformat{false}; // -E
  bool expandIncludeLinesInPreprocessedOutput{true};
  bool showColors{false};
};

} // namespace Fortran::parser

#endif // FORTRAN_PARSER_OPTIONS_H_
