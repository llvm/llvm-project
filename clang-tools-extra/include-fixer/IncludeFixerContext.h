//===-- IncludeFixerContext.h - Include fixer context -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INCLUDEFIXERCONTEXT_H
#define LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INCLUDEFIXERCONTEXT_H

#include <string>
#include <vector>

namespace clang {
namespace include_fixer {

/// \brief A context for the symbol being queried.
struct IncludeFixerContext {
  /// \brief The symbol name.
  std::string SymbolIdentifer;
  /// \brief The headers which have SymbolIdentifier definitions.
  std::vector<std::string> Headers;
};

} // namespace include_fixer
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INCLUDEFIXERCONTEXT_H
