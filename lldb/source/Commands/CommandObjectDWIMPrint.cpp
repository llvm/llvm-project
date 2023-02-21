//===-- CommandObjectDWIMPrint.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectDWIMPrint.h"

#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/DumpValueObjectOptions.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/OptionGroupFormat.h"
#include "lldb/Interpreter/OptionGroupValueObjectDisplay.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_private;

CommandObjectDWIMPrint::CommandObjectDWIMPrint(CommandInterpreter &interpreter)
    : CommandObjectRaw(interpreter, "dwim-print",
                       "Print a variable or expression.",
                       "dwim-print [<variable-name> | <expression>]",
                       eCommandProcessMustBePaused | eCommandTryTargetAPILock) {
  m_option_group.Append(&m_format_options,
                        OptionGroupFormat::OPTION_GROUP_FORMAT |
                            OptionGroupFormat::OPTION_GROUP_GDB_FMT,
                        LLDB_OPT_SET_1);
  StringRef exclude_expr_options[] = {"debug", "top-level"};
  m_option_group.Append(&m_expr_options, exclude_expr_options);
  m_option_group.Append(&m_varobj_options, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1);
  m_option_group.Finalize();
}

Options *CommandObjectDWIMPrint::GetOptions() { return &m_option_group; }

bool CommandObjectDWIMPrint::DoExecute(StringRef command,
                                       CommandReturnObject &result) {
  m_option_group.NotifyOptionParsingStarting(&m_exe_ctx);
  OptionsWithRaw args{command};
  StringRef expr = args.GetRawPart();

  if (args.HasArgs()) {
    if (!ParseOptionsAndNotify(args.GetArgs(), result, m_option_group,
                               m_exe_ctx))
      return false;
  } else if (command.empty()) {
    result.AppendErrorWithFormatv("'{0}' takes a variable or expression",
                                  m_cmd_name);
    return false;
  }

  auto verbosity = GetDebugger().GetDWIMPrintVerbosity();

  DumpValueObjectOptions dump_options = m_varobj_options.GetAsDumpOptions(
      m_expr_options.m_verbosity, m_format_options.GetFormat());

  // First, try `expr` as the name of a frame variable.
  if (StackFrame *frame = m_exe_ctx.GetFramePtr()) {
    auto valobj_sp = frame->FindVariable(ConstString(expr));
    if (valobj_sp && valobj_sp->GetError().Success()) {
      if (verbosity == eDWIMPrintVerbosityFull) {
        StringRef flags;
        if (args.HasArgs())
          flags = args.GetArgString();
        result.AppendMessageWithFormatv("note: ran `frame variable {0}{1}`",
                                        flags, expr);
      }
      valobj_sp->Dump(result.GetOutputStream(), dump_options);
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
    const EvaluateExpressionOptions eval_options =
        m_expr_options.GetEvaluateExpressionOptions(target, m_varobj_options);
    ValueObjectSP valobj_sp;
    ExpressionResults expr_result =
        target.EvaluateExpression(expr, exe_scope, valobj_sp, eval_options);
    if (expr_result == eExpressionCompleted) {
      if (verbosity != eDWIMPrintVerbosityNone) {
        StringRef flags;
        if (args.HasArgs())
          flags = args.GetArgStringWithDelimiter();
        result.AppendMessageWithFormatv("note: ran `expression {0}{1}`", flags,
                                        expr);
      }
      valobj_sp->Dump(result.GetOutputStream(), dump_options);
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
