//===--- CheckerOptInfo.h - Specifies which checkers to use -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_CHECKEROPTINFO_H
#define LLVM_CLANG_STATICANALYZER_CORE_CHECKEROPTINFO_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace ento {

/// Represents a request to include or exclude a checker or package from a
/// specific analysis run.
///
/// \sa CheckerRegistry::initializeManager
class CheckerOptInfo {
  StringRef Name;
  bool Enable;
  bool Claimed;

public:
  CheckerOptInfo(StringRef name, bool enable)
    : Name(name), Enable(enable), Claimed(false) { }

  StringRef getName() const { return Name; }
  bool isEnabled() const { return Enable; }
  bool isDisabled() const { return !isEnabled(); }

  bool isClaimed() const { return Claimed; }
  bool isUnclaimed() const { return !isClaimed(); }
  void claim() { Claimed = true; }
};

} // end namespace ento
} // end namespace clang

#endif
