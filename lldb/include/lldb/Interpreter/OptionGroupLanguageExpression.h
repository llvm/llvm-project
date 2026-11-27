//===-- OptionGroupLanguageExpression.h -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_OPTIONGROUPLANGUAGEEXPRESSION_H
#define LLDB_INTERPRETER_OPTIONGROUPLANGUAGEEXPRESSION_H

#include "lldb/Interpreter/Options.h"

namespace lldb_private {

class EvaluateExpressionOptions;

/// The options that a language plugin adds to the commands that evaluate
/// expressions.
class OptionGroupLanguageExpression : public OptionGroup {
public:
  /// Stores the values of the options in \p options.
  virtual void Apply(EvaluateExpressionOptions &options) const = 0;
};

} // namespace lldb_private

#endif // LLDB_INTERPRETER_OPTIONGROUPLANGUAGEEXPRESSION_H
