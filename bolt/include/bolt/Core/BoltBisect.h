//===- llvm/IR/BoltBisect.h - LLVM Bisect support ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the interface for bisecting optimizations.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_OPTBISECT_H
#define LLVM_IR_OPTBISECT_H

#include "llvm/ADT/StringRef.h"
#include <limits>

namespace llvm {

class BoltPassGate {
public:
  virtual ~BoltPassGate() = default;

  virtual bool shouldRunPass(const StringRef PassName) {
    return true;
  }

  virtual bool isEnabled() const { return false; }
};

class BoltBisect : public BoltPassGate {
public:

  BoltBisect() = default;

  virtual ~BoltBisect() = default;

  bool shouldRunPass(const StringRef PassName) override;

  bool isEnabled() const override { return BisectLimit != Disabled; }

  void setLimit(int Limit) {
    BisectLimit = Limit;
    LastBisectNum = 0;
  }

  static const int Disabled = std::numeric_limits<int>::max();

private:
  int BisectLimit = Disabled;
  int LastBisectNum = 0;
};

BoltPassGate &getGlobalPassGate();

} // end namespace llvm

#endif // LLVM_IR_OPTBISECT_H
