//===-- PathConfig.h - Process paths of symbols -----------------*- C++ -*-===//
//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_PATH_CONFIG_H
#define LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_PATH_CONFIG_H

#include "HeaderMapCollector.h"
#include "clang/Basic/SourceManager.h"
#include <string>

namespace clang {
namespace find_all_symbols {

/// \brief This calculates the include path for \p Loc.
///
/// \param SM SourceManager.
/// \param Loc A SourceLocation.
/// \param Collector An optional header mapping collector.
///
/// \return The file path (or mapped file path if Collector is provided) of the
/// header that includes \p Loc. If \p Loc comes from .inc header file, \p Loc
/// is set to the location from which the .inc header file is included. If \p
/// Loc is invalid or comes from a main file, this returns an empty string.
std::string getIncludePath(const SourceManager &SM, SourceLocation Loc,
                           const HeaderMapCollector *Collector = nullptr);

} // namespace find_all_symbols
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_PATH_CONFIG_H
