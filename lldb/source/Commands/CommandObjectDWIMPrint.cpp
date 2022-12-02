//===-- CommandObjectDWIMPrint.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectDWIMPrint.h"

#include "lldb/Core/ValueObject.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_private;

CommandObjectDWIMPrint::CommandObjectDWIMPrint(CommandInterpreter &interpreter)
    : CommandObjectRaw(interpreter, "dwim-print",
                       "Print a variable or expression.",
                       "dwim-print [<variable-name> | <expression>]",
                       eCommandProcessMustBePaused | eCommandTryTargetAPILock) {
}

bool CommandObjectDWIMPrint::DoExecute(StringRef expr,
                                       CommandReturnObject &result) {
  // Ignore leading and trailing whitespace.
  expr = expr.trim();

  if (expr.empty()) {
    result.AppendErrorWithFormatv("'{0}' takes a variable or expression",
                                  m_cmd_name);
    return false;
  }

  auto verbosity = GetDebugger().GetDWIMPrintVerbosity();

  // First, try `expr` as the name of a frame variable.
  if (StackFrame *frame = m_exe_ctx.GetFramePtr()) {
    auto valobj_sp = frame->FindVariable(ConstString(expr));
    if (valobj_sp && valobj_sp->GetError().Success()) {
      if (verbosity == eDWIMPrintVerbosityFull)
        result.AppendMessageWithFormatv("note: ran `frame variable {0}`", expr);
      valobj_sp->Dump(result.GetOutputStream());
      result.SetStatus(eReturnStatusSuccessFinishResult);
      return true;
    }
  }

  // Second, also lastly, try `expr` as a source expression to evaluate.
  {
    Target *target_ptr = m_exe_ctx.GetTargetPtr();
    // Fallback to the dummy target, which can allow for expression evaluation.
    Target &target = target_ptr ? *target_ptr : GetDummyTarget();

    auto *exe_scope = m_exe_ctx.GetBestExecutionContextScope();
    ValueObjectSP valobj_sp;
    if (target.EvaluateExpression(expr, exe_scope, valobj_sp) ==
        eExpressionCompleted) {
      if (verbosity != eDWIMPrintVerbosityNone)
        result.AppendMessageWithFormatv("note: ran `expression -- {0}`", expr);
      valobj_sp->Dump(result.GetOutputStream());
      result.SetStatus(eReturnStatusSuccessFinishResult);
      return true;
    } else {
      if (valobj_sp)
        result.SetError(valobj_sp->GetError());
      else
        result.AppendErrorWithFormatv(
            "unknown error evaluating expression `{0}`", expr);
      return false;
    }
  }
}
