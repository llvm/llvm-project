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
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-types.h"
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
  CommandCompletions::InvokeCommonCompletionCallbacks(
      GetCommandInterpreter(), CommandCompletions::eVariablePathCompletion,
      request, nullptr);
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

  auto verbosity = GetDebugger().GetDWIMPrintVerbosity();

  Target *target_ptr = m_exe_ctx.GetTargetPtr();
  // Fallback to the dummy target, which can allow for expression evaluation.
  Target &target = target_ptr ? *target_ptr : GetDummyTarget();

  const EvaluateExpressionOptions eval_options =
      m_expr_options.GetEvaluateExpressionOptions(target, m_varobj_options);

  DumpValueObjectOptions dump_options = m_varobj_options.GetAsDumpOptions(
      m_expr_options.m_verbosity, m_format_options.GetFormat());
  dump_options.SetHideRootName(eval_options.GetSuppressPersistentResult());

  StackFrame *frame = m_exe_ctx.GetFramePtr();

  // First, try `expr` as the name of a frame variable.
  if (frame) {
    auto valobj_sp = frame->FindVariable(ConstString(expr));
    if (valobj_sp && valobj_sp->GetError().Success()) {
      if (!eval_options.GetSuppressPersistentResult()) {
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

  // For Swift frames, rewrite `po 0x12345600` to use `unsafeBitCast`.
  //
  // This works only when the address points to an instance of a class. This
  // matches the behavior of `po` in Objective-C frames.
  //
  // The following conditions are required:
  //   1. The command is `po` (or equivalently the `-O` flag is used)
  //   2. The current language is Swift
  //   3. The expression is entirely a integer value (decimal or hex)
  //   4. The integer passes sanity checks as a memory address
  //
  // The address sanity checks are:
  //   1. The integer represents a readable memory address
  //
  // Future potential sanity checks:
  //   1. Accept tagged pointers/values
  //   2. Verify the isa pointer is a known class
  //   3. Require addresses to be on the heap
  std::string modified_expr_storage;
  // Either Swift was explicitly specified, or the frame is Swift.
  bool is_swift = false;
  if (m_expr_options.language == lldb::eLanguageTypeSwift)
    is_swift = true;
  else if (m_expr_options.language == lldb::eLanguageTypeUnknown)
    is_swift = frame && frame->GuessLanguage() == lldb::eLanguageTypeSwift;
  bool is_po = m_varobj_options.use_objc;
  if (is_swift && is_po) {
    lldb::addr_t addr;
    bool is_integer = !expr.getAsInteger(0, addr);
    if (is_integer) {
      MemoryRegionInfo mem_info;
      m_exe_ctx.GetProcessRef().GetMemoryRegionInfo(addr, mem_info);
      bool is_readable = mem_info.GetReadable() == MemoryRegionInfo::eYes;
      if (is_readable) {
        modified_expr_storage =
            llvm::formatv("unsafeBitCast({0}, to: AnyObject.self)", expr).str();
        expr = modified_expr_storage;
      }
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
