//===-- CallFrameInfo.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SYMBOL_CALLFRAMEINFO_H
#define LLDB_SYMBOL_CALLFRAMEINFO_H

#include "lldb/Core/Address.h"

namespace lldb_private {

class CallFrameInfo {
public:
  virtual ~CallFrameInfo() = default;

  virtual bool GetAddressRange(Address addr, AddressRange &range) = 0;

  virtual std::unique_ptr<UnwindPlan>
  GetUnwindPlan(llvm::ArrayRef<AddressRange> ranges, const Address &addr) = 0;

  virtual std::unique_ptr<UnwindPlan> GetUnwindPlan(const Address &addr) = 0;
};

} // namespace lldb_private

#endif // LLDB_SYMBOL_CALLFRAMEINFO_H
