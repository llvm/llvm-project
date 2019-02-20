//===-- OptionValueArgs.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionValueArgs_h_
#define liblldb_OptionValueArgs_h_

#include "lldb/Interpreter/OptionValueArray.h"

namespace lldb_private {

class OptionValueArgs : public OptionValueArray {
public:
  OptionValueArgs()
      : OptionValueArray(
            OptionValue::ConvertTypeToMask(OptionValue::eTypeString)) {}

  ~OptionValueArgs() override {}

  size_t GetArgs(Args &args);

  Type GetType() const override { return eTypeArgs; }
};

} // namespace lldb_private

#endif // liblldb_OptionValueArgs_h_
