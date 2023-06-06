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
#include "lldb/Expression/ExpressionVariable.h"
#include "lldb/Expression/UserExpression.h"
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

  CommandArgumentData var_name_arg(eArgTypeVarName, eArgRepeatPlain);
  m_arguments.push_back({var_name_arg});

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

void CommandObjectDWIMPrint::HandleArgumentCompletion(
    CompletionRequest &request, OptionElementVector &opt_element_vector) {
  lldb_private::CommandCompletions::InvokeCommonCompletionCallbacks(
      GetCommandInterpreter(), lldb::eVariablePathCompletion, request, nullptr);
}

bool CommandObjectDWIMPrint::DoExecute(StringRef command,
                                       CommandReturnObject &result) {
  m_option_group.NotifyOptionParsingStarting(&m_exe_ctx);
  OptionsWithRaw args{command};
  StringRef expr = args.GetRawPart();

  if (expr.empty()) {
    result.AppendErrorWithFormatv("'{0}' takes a variable or expression",
                                  m_cmd_name);
    return false;
  }

  if (args.HasArgs()) {
    if (!ParseOptionsAndNotify(args.GetArgs(), result, m_option_group,
                               m_exe_ctx))
      return false;
  }

  // If the user has not specified, default to disabling persistent results.
  if (m_expr_options.suppress_persistent_result == eLazyBoolCalculate)
    m_expr_options.suppress_persistent_result = eLazyBoolYes;
  bool suppress_result = m_expr_options.ShouldSuppressResult(m_varobj_options);

  auto verbosity = GetDebugger().GetDWIMPrintVerbosity();

  Target *target_ptr = m_exe_ctx.GetTargetPtr();
  // Fallback to the dummy target, which can allow for expression evaluation.
  Target &target = target_ptr ? *target_ptr : GetDummyTarget();

  EvaluateExpressionOptions eval_options =
      m_expr_options.GetEvaluateExpressionOptions(target, m_varobj_options);
  // This command manually removes the result variable, make sure expression
  // evaluation doesn't do it first.
  eval_options.SetSuppressPersistentResult(false);

  DumpValueObjectOptions dump_options = m_varobj_options.GetAsDumpOptions(
      m_expr_options.m_verbosity, m_format_options.GetFormat());
  dump_options.SetHideRootName(suppress_result);

  StackFrame *frame = m_exe_ctx.GetFramePtr();

  // First, try `expr` as the name of a frame variable.
  if (frame) {
    auto valobj_sp = frame->FindVariable(ConstString(expr));
    if (valobj_sp && valobj_sp->GetError().Success()) {
      if (!suppress_result) {
        if (auto persisted_valobj = valobj_sp->Persist())
          valobj_sp = persisted_valobj;
      }

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
    auto *exe_scope = m_exe_ctx.GetBestExecutionContextScope();
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

      if (valobj_sp->GetError().GetError() != UserExpression::kNoResult)
        valobj_sp->Dump(result.GetOutputStream(), dump_options);

      if (suppress_result)
        if (auto result_var_sp =
                target.GetPersistentVariable(valobj_sp->GetName())) {
          auto language = valobj_sp->GetPreferredDisplayLanguage();
          if (auto *persistent_state =
                  target.GetPersistentExpressionStateForLanguage(language))
            persistent_state->RemovePersistentVariable(result_var_sp);
        }

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
