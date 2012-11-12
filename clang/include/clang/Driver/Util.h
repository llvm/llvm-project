//===--- Util.h - Common Driver Utilities -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_UTIL_H_
#define CLANG_DRIVER_UTIL_H_

#include "clang/Basic/LLVM.h"

namespace clang {
namespace driver {
  class Action;

  /// ArgStringList - Type used for constructing argv lists for subprocesses.
  typedef SmallVector<const char*, 16> ArgStringList;

  /// ActionList - Type used for lists of actions.
  typedef SmallVector<Action*, 3> ActionList;

} // end namespace driver
} // end namespace clang

#endif
