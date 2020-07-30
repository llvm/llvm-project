//===-- SBCommandInterpreter.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-types.h"

#include "SBReproducerPrivate.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Listener.h"

#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandInterpreterRunOptions.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBExecutionContext.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStringList.h"
#include "lldb/API/SBTarget.h"

#include <memory>

using namespace lldb;
using namespace lldb_private;

class CommandPluginInterfaceImplementation : public CommandObjectParsed {
public:
  CommandPluginInterfaceImplementation(CommandInterpreter &interpreter,
                                       const char *name,
                                       lldb::SBCommandPluginInterface *backend,
                                       const char *help = nullptr,
                                       const char *syntax = nullptr,
                                       uint32_t flags = 0,
                                       const char *auto_repeat_command = "")
      : CommandObjectParsed(interpreter, name, help, syntax, flags),
        m_backend(backend) {
    m_auto_repeat_command =
        auto_repeat_command == nullptr
            ? llvm::None
            : llvm::Optional<std::string>(auto_repeat_command);
  }

  bool IsRemovable() const override { return true; }

  /// More documentation is available in lldb::CommandObject::GetRepeatCommand,
  /// but in short, if nullptr is returned, the previous command will be
  /// repeated, and if an empty string is returned, no commands will be
  /// executed.
  const char *GetRepeatCommand(Args &current_command_args,
                               uint32_t index) override {
    if (!m_auto_repeat_command)
      return nullptr;
    else
      return m_auto_repeat_command->c_str();
  }

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    SBCommandReturnObject sb_return(result);
    SBCommandInterpreter sb_interpreter(&m_interpreter);
    SBDebugger debugger_sb(m_interpreter.GetDebugger().shared_from_this());
    bool ret = m_backend->DoExecute(
        debugger_sb, command.GetArgumentVector(), sb_return);
    return ret;
  }
  std::shared_ptr<lldb::SBCommandPluginInterface> m_backend;
  llvm::Optional<std::string> m_auto_repeat_command;
};

SBCommandInterpreter::SBCommandInterpreter(CommandInterpreter *interpreter)
    : m_opaque_ptr(interpreter) {
  LLDB_RECORD_CONSTRUCTOR(SBCommandInterpreter,
                          (lldb_private::CommandInterpreter *), interpreter);

}

SBCommandInterpreter::SBCommandInterpreter(const SBCommandInterpreter &rhs)
    : m_opaque_ptr(rhs.m_opaque_ptr) {
  LLDB_RECORD_CONSTRUCTOR(SBCommandInterpreter,
                          (const lldb::SBCommandInterpreter &), rhs);
}

SBCommandInterpreter::~SBCommandInterpreter() = default;

const SBCommandInterpreter &SBCommandInterpreter::
operator=(const SBCommandInterpreter &rhs) {
  LLDB_RECORD_METHOD(
      const lldb::SBCommandInterpreter &,
      SBCommandInterpreter, operator=,(const lldb::SBCommandInterpreter &),
      rhs);

  m_opaque_ptr = rhs.m_opaque_ptr;
  return LLDB_RECORD_RESULT(*this);
}

bool SBCommandInterpreter::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBCommandInterpreter, IsValid);
  return this->operator bool();
}
SBCommandInterpreter::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBCommandInterpreter, operator bool);

  return m_opaque_ptr != nullptr;
}

bool SBCommandInterpreter::CommandExists(const char *cmd) {
  LLDB_RECORD_METHOD(bool, SBCommandInterpreter, CommandExists, (const char *),
                     cmd);

  return (((cmd != nullptr) && IsValid()) ? m_opaque_ptr->CommandExists(cmd)
                                          : false);
}

bool SBCommandInterpreter::AliasExists(const char *cmd) {
  LLDB_RECORD_METHOD(bool, SBCommandInterpreter, AliasExists, (const char *),
                     cmd);

  return (((cmd != nullptr) && IsValid()) ? m_opaque_ptr->AliasExists(cmd)
                                          : false);
}

bool SBCommandInterpreter::IsActive() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBCommandInterpreter, IsActive);

  return (IsValid() ? m_opaque_ptr->IsActive() : false);
}

bool SBCommandInterpreter::WasInterrupted() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBCommandInterpreter, WasInterrupted);

  return (IsValid() ? m_opaque_ptr->WasInterrupted() : false);
}

const char *SBCommandInterpreter::GetIOHandlerControlSequence(char ch) {
  LLDB_RECORD_METHOD(const char *, SBCommandInterpreter,
                     GetIOHandlerControlSequence, (char), ch);

  return (IsValid()
              ? m_opaque_ptr->GetDebugger()
                    .GetTopIOHandlerControlSequence(ch)
                    .GetCString()
              : nullptr);
}

lldb::ReturnStatus
SBCommandInterpreter::HandleCommand(const char *command_line,
                                    SBCommandReturnObject &result,
                                    bool add_to_history) {
  LLDB_RECORD_METHOD(lldb::ReturnStatus, SBCommandInterpreter, HandleCommand,
                     (const char *, lldb::SBCommandReturnObject &, bool),
                     command_line, result, add_to_history);

  SBExecutionContext sb_exe_ctx;
  return HandleCommand(command_line, sb_exe_ctx, result, add_to_history);
}

lldb::ReturnStatus SBCommandInterpreter::HandleCommand(
    const char *command_line, SBExecutionContext &override_context,
    SBCommandReturnObject &result, bool add_to_history) {
  LLDB_RECORD_METHOD(lldb::ReturnStatus, SBCommandInterpreter, HandleCommand,
                     (const char *, lldb::SBExecutionContext &,
                      lldb::SBCommandReturnObject &, bool),
                     command_line, override_context, result, add_to_history);


  ExecutionContext ctx, *ctx_ptr;
  if (override_context.get()) {
    ctx = override_context.get()->Lock(true);
    ctx_ptr = &ctx;
  } else
    ctx_ptr = nullptr;

  result.Clear();
  if (command_line && IsValid()) {
    result.ref().SetInteractive(false);
    m_opaque_ptr->HandleCommand(command_line,
                                add_to_history ? eLazyBoolYes : eLazyBoolNo,
                                result.ref(), ctx_ptr);
  } else {
    result->AppendError(
        "SBCommandInterpreter or the command line is not valid");
    result->SetStatus(eReturnStatusFailed);
  }


  return result.GetStatus();
}

void SBCommandInterpreter::HandleCommandsFromFile(
    lldb::SBFileSpec &file, lldb::SBExecutionContext &override_context,
    lldb::SBCommandInterpreterRunOptions &options,
    lldb::SBCommandReturnObject result) {
  LLDB_RECORD_METHOD(void, SBCommandInterpreter, HandleCommandsFromFile,
                     (lldb::SBFileSpec &, lldb::SBExecutionContext &,
                      lldb::SBCommandInterpreterRunOptions &,
                      lldb::SBCommandReturnObject),
                     file, override_context, options, result);

  if (!IsValid()) {
    result->AppendError("SBCommandInterpreter is not valid.");
    result->SetStatus(eReturnStatusFailed);
    return;
  }

  if (!file.IsValid()) {
    SBStream s;
    file.GetDescription(s);
    result->AppendErrorWithFormat("File is not valid: %s.", s.GetData());
    result->SetStatus(eReturnStatusFailed);
  }

  FileSpec tmp_spec = file.ref();
  ExecutionContext ctx, *ctx_ptr;
  if (override_context.get()) {
    ctx = override_context.get()->Lock(true);
    ctx_ptr = &ctx;
  } else
    ctx_ptr = nullptr;

  m_opaque_ptr->HandleCommandsFromFile(tmp_spec, ctx_ptr, options.ref(),
                                       result.ref());
}

int SBCommandInterpreter::HandleCompletion(
    const char *current_line, const char *cursor, const char *last_char,
    int match_start_point, int max_return_elements, SBStringList &matches) {
  LLDB_RECORD_METHOD(int, SBCommandInterpreter, HandleCompletion,
                     (const char *, const char *, const char *, int, int,
                      lldb::SBStringList &),
                     current_line, cursor, last_char, match_start_point,
                     max_return_elements, matches);

  SBStringList dummy_descriptions;
  return HandleCompletionWithDescriptions(
      current_line, cursor, last_char, match_start_point, max_return_elements,
      matches, dummy_descriptions);
}

int SBCommandInterpreter::HandleCompletionWithDescriptions(
    const char *current_line, const char *cursor, const char *last_char,
    int match_start_point, int max_return_elements, SBStringList &matches,
    SBStringList &descriptions) {
  LLDB_RECORD_METHOD(int, SBCommandInterpreter,
                     HandleCompletionWithDescriptions,
                     (const char *, const char *, const char *, int, int,
                      lldb::SBStringList &, lldb::SBStringList &),
                     current_line, cursor, last_char, match_start_point,
                     max_return_elements, matches, descriptions);

  // Sanity check the arguments that are passed in: cursor & last_char have to
  // be within the current_line.
  if (current_line == nullptr || cursor == nullptr || last_char == nullptr)
    return 0;

  if (cursor < current_line || last_char < current_line)
    return 0;

  size_t current_line_size = strlen(current_line);
  if (cursor - current_line > static_cast<ptrdiff_t>(current_line_size) ||
      last_char - current_line > static_cast<ptrdiff_t>(current_line_size))
    return 0;

  if (!IsValid())
    return 0;

  lldb_private::StringList lldb_matches, lldb_descriptions;
  CompletionResult result;
  CompletionRequest request(current_line, cursor - current_line, result);
  m_opaque_ptr->HandleCompletion(request);
  result.GetMatches(lldb_matches);
  result.GetDescriptions(lldb_descriptions);

  // Make the result array indexed from 1 again by adding the 'common prefix'
  // of all completions as element 0. This is done to emulate the old API.
  if (request.GetParsedLine().GetArgumentCount() == 0) {
    // If we got an empty string, insert nothing.
    lldb_matches.InsertStringAtIndex(0, "");
    lldb_descriptions.InsertStringAtIndex(0, "");
  } else {
    // Now figure out if there is a common substring, and if so put that in
    // element 0, otherwise put an empty string in element 0.
    std::string command_partial_str = request.GetCursorArgumentPrefix().str();

    std::string common_prefix = lldb_matches.LongestCommonPrefix();
    const size_t partial_name_len = command_partial_str.size();
    common_prefix.erase(0, partial_name_len);

    // If we matched a unique single command, add a space... Only do this if
    // the completer told us this was a complete word, however...
    if (lldb_matches.GetSize() == 1) {
      char quote_char = request.GetParsedArg().GetQuoteChar();
      common_prefix =
          Args::EscapeLLDBCommandArgument(common_prefix, quote_char);
      if (request.GetParsedArg().IsQuoted())
        common_prefix.push_back(quote_char);
      common_prefix.push_back(' ');
    }
    lldb_matches.InsertStringAtIndex(0, common_prefix.c_str());
    lldb_descriptions.InsertStringAtIndex(0, "");
  }

  SBStringList temp_matches_list(&lldb_matches);
  matches.AppendList(temp_matches_list);
  SBStringList temp_descriptions_list(&lldb_descriptions);
  descriptions.AppendList(temp_descriptions_list);
  return result.GetNumberOfResults();
}

int SBCommandInterpreter::HandleCompletionWithDescriptions(
    const char *current_line, uint32_t cursor_pos, int match_start_point,
    int max_return_elements, SBStringList &matches,
    SBStringList &descriptions) {
  LLDB_RECORD_METHOD(int, SBCommandInterpreter,
                     HandleCompletionWithDescriptions,
                     (const char *, uint32_t, int, int, lldb::SBStringList &,
                      lldb::SBStringList &),
                     current_line, cursor_pos, match_start_point,
                     max_return_elements, matches, descriptions);

  const char *cursor = current_line + cursor_pos;
  const char *last_char = current_line + strlen(current_line);
  return HandleCompletionWithDescriptions(
      current_line, cursor, last_char, match_start_point, max_return_elements,
      matches, descriptions);
}

int SBCommandInterpreter::HandleCompletion(const char *current_line,
                                           uint32_t cursor_pos,
                                           int match_start_point,
                                           int max_return_elements,
                                           lldb::SBStringList &matches) {
  LLDB_RECORD_METHOD(int, SBCommandInterpreter, HandleCompletion,
                     (const char *, uint32_t, int, int, lldb::SBStringList &),
                     current_line, cursor_pos, match_start_point,
                     max_return_elements, matches);

  const char *cursor = current_line + cursor_pos;
  const char *last_char = current_line + strlen(current_line);
  return HandleCompletion(current_line, cursor, last_char, match_start_point,
                          max_return_elements, matches);
}

bool SBCommandInterpreter::HasCommands() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBCommandInterpreter, HasCommands);

  return (IsValid() ? m_opaque_ptr->HasCommands() : false);
}

bool SBCommandInterpreter::HasAliases() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBCommandInterpreter, HasAliases);

  return (IsValid() ? m_opaque_ptr->HasAliases() : false);
}

bool SBCommandInterpreter::HasAliasOptions() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBCommandInterpreter, HasAliasOptions);

  return (IsValid() ? m_opaque_ptr->HasAliasOptions() : false);
}

SBProcess SBCommandInterpreter::GetProcess() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBProcess, SBCommandInterpreter, GetProcess);

  SBProcess sb_process;
  ProcessSP process_sp;
  if (IsValid()) {
    TargetSP target_sp(m_opaque_ptr->GetDebugger().GetSelectedTarget());
    if (target_sp) {
      std::lock_guard<std::recursive_mutex> guard(target_sp->GetAPIMutex());
      process_sp = target_sp->GetProcessSP();
      sb_process.SetSP(process_sp);
    }
  }

  return LLDB_RECORD_RESULT(sb_process);
}

SBDebugger SBCommandInterpreter::GetDebugger() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBDebugger, SBCommandInterpreter,
                             GetDebugger);

  SBDebugger sb_debugger;
  if (IsValid())
    sb_debugger.reset(m_opaque_ptr->GetDebugger().shared_from_this());

  return LLDB_RECORD_RESULT(sb_debugger);
}

bool SBCommandInterpreter::GetPromptOnQuit() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBCommandInterpreter, GetPromptOnQuit);

  return (IsValid() ? m_opaque_ptr->GetPromptOnQuit() : false);
}

void SBCommandInterpreter::SetPromptOnQuit(bool b) {
  LLDB_RECORD_METHOD(void, SBCommandInterpreter, SetPromptOnQuit, (bool), b);

  if (IsValid())
    m_opaque_ptr->SetPromptOnQuit(b);
}

void SBCommandInterpreter::AllowExitCodeOnQuit(bool allow) {
  LLDB_RECORD_METHOD(void, SBCommandInterpreter, AllowExitCodeOnQuit, (bool),
                     allow);

  if (m_opaque_ptr)
    m_opaque_ptr->AllowExitCodeOnQuit(allow);
}

bool SBCommandInterpreter::HasCustomQuitExitCode() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBCommandInterpreter, HasCustomQuitExitCode);

  bool exited = false;
  if (m_opaque_ptr)
    m_opaque_ptr->GetQuitExitCode(exited);
  return exited;
}

int SBCommandInterpreter::GetQuitStatus() {
  LLDB_RECORD_METHOD_NO_ARGS(int, SBCommandInterpreter, GetQuitStatus);

  bool exited = false;
  return (m_opaque_ptr ? m_opaque_ptr->GetQuitExitCode(exited) : 0);
}

void SBCommandInterpreter::ResolveCommand(const char *command_line,
                                          SBCommandReturnObject &result) {
  LLDB_RECORD_METHOD(void, SBCommandInterpreter, ResolveCommand,
                     (const char *, lldb::SBCommandReturnObject &),
                     command_line, result);

  result.Clear();
  if (command_line && IsValid()) {
    m_opaque_ptr->ResolveCommand(command_line, result.ref());
  } else {
    result->AppendError(
        "SBCommandInterpreter or the command line is not valid");
    result->SetStatus(eReturnStatusFailed);
  }
}

CommandInterpreter *SBCommandInterpreter::get() { return m_opaque_ptr; }

CommandInterpreter &SBCommandInterpreter::ref() {
  assert(m_opaque_ptr);
  return *m_opaque_ptr;
}

void SBCommandInterpreter::reset(
    lldb_private::CommandInterpreter *interpreter) {
  m_opaque_ptr = interpreter;
}

void SBCommandInterpreter::SourceInitFileInHomeDirectory(
    SBCommandReturnObject &result) {
  LLDB_RECORD_METHOD(void, SBCommandInterpreter, SourceInitFileInHomeDirectory,
                     (lldb::SBCommandReturnObject &), result);

  result.Clear();
  if (IsValid()) {
    TargetSP target_sp(m_opaque_ptr->GetDebugger().GetSelectedTarget());
    std::unique_lock<std::recursive_mutex> lock;
    if (target_sp)
      lock = std::unique_lock<std::recursive_mutex>(target_sp->GetAPIMutex());
    m_opaque_ptr->SourceInitFileHome(result.ref());
  } else {
    result->AppendError("SBCommandInterpreter is not valid");
    result->SetStatus(eReturnStatusFailed);
  }
}

void SBCommandInterpreter::SourceInitFileInCurrentWorkingDirectory(
    SBCommandReturnObject &result) {
  LLDB_RECORD_METHOD(void, SBCommandInterpreter,
                     SourceInitFileInCurrentWorkingDirectory,
                     (lldb::SBCommandReturnObject &), result);

  result.Clear();
  if (IsValid()) {
    TargetSP target_sp(m_opaque_ptr->GetDebugger().GetSelectedTarget());
    std::unique_lock<std::recursive_mutex> lock;
    if (target_sp)
      lock = std::unique_lock<std::recursive_mutex>(target_sp->GetAPIMutex());
    m_opaque_ptr->SourceInitFileCwd(result.ref());
  } else {
    result->AppendError("SBCommandInterpreter is not valid");
    result->SetStatus(eReturnStatusFailed);
  }
}

SBBroadcaster SBCommandInterpreter::GetBroadcaster() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBBroadcaster, SBCommandInterpreter,
                             GetBroadcaster);


  SBBroadcaster broadcaster(m_opaque_ptr, false);


  return LLDB_RECORD_RESULT(broadcaster);
}

const char *SBCommandInterpreter::GetBroadcasterClass() {
  LLDB_RECORD_STATIC_METHOD_NO_ARGS(const char *, SBCommandInterpreter,
                                    GetBroadcasterClass);

  return CommandInterpreter::GetStaticBroadcasterClass().AsCString();
}

const char *SBCommandInterpreter::GetArgumentTypeAsCString(
    const lldb::CommandArgumentType arg_type) {
  LLDB_RECORD_STATIC_METHOD(const char *, SBCommandInterpreter,
                            GetArgumentTypeAsCString,
                            (const lldb::CommandArgumentType), arg_type);

  return CommandObject::GetArgumentTypeAsCString(arg_type);
}

const char *SBCommandInterpreter::GetArgumentDescriptionAsCString(
    const lldb::CommandArgumentType arg_type) {
  LLDB_RECORD_STATIC_METHOD(const char *, SBCommandInterpreter,
                            GetArgumentDescriptionAsCString,
                            (const lldb::CommandArgumentType), arg_type);

  return CommandObject::GetArgumentDescriptionAsCString(arg_type);
}

bool SBCommandInterpreter::EventIsCommandInterpreterEvent(
    const lldb::SBEvent &event) {
  LLDB_RECORD_STATIC_METHOD(bool, SBCommandInterpreter,
                            EventIsCommandInterpreterEvent,
                            (const lldb::SBEvent &), event);

  return event.GetBroadcasterClass() ==
         SBCommandInterpreter::GetBroadcasterClass();
}

bool SBCommandInterpreter::SetCommandOverrideCallback(
    const char *command_name, lldb::CommandOverrideCallback callback,
    void *baton) {
  LLDB_RECORD_DUMMY(bool, SBCommandInterpreter, SetCommandOverrideCallback,
                    (const char *, lldb::CommandOverrideCallback, void *),
                    command_name, callback, baton);

  if (command_name && command_name[0] && IsValid()) {
    llvm::StringRef command_name_str = command_name;
    CommandObject *cmd_obj =
        m_opaque_ptr->GetCommandObjectForCommand(command_name_str);
    if (cmd_obj) {
      assert(command_name_str.empty());
      cmd_obj->SetOverrideCallback(callback, baton);
      return true;
    }
  }
  return false;
}

lldb::SBCommand SBCommandInterpreter::AddMultiwordCommand(const char *name,
                                                          const char *help) {
  LLDB_RECORD_METHOD(lldb::SBCommand, SBCommandInterpreter, AddMultiwordCommand,
                     (const char *, const char *), name, help);

  CommandObjectMultiword *new_command =
      new CommandObjectMultiword(*m_opaque_ptr, name, help);
  new_command->SetRemovable(true);
  lldb::CommandObjectSP new_command_sp(new_command);
  if (new_command_sp &&
      m_opaque_ptr->AddUserCommand(name, new_command_sp, true))
    return LLDB_RECORD_RESULT(lldb::SBCommand(new_command_sp));
  return LLDB_RECORD_RESULT(lldb::SBCommand());
}

lldb::SBCommand SBCommandInterpreter::AddCommand(
    const char *name, lldb::SBCommandPluginInterface *impl, const char *help) {
  LLDB_RECORD_METHOD(
      lldb::SBCommand, SBCommandInterpreter, AddCommand,
      (const char *, lldb::SBCommandPluginInterface *, const char *), name,
      impl, help);

  return LLDB_RECORD_RESULT(AddCommand(name, impl, help, /*syntax=*/nullptr,
                                       /*auto_repeat_command=*/""))
}

lldb::SBCommand
SBCommandInterpreter::AddCommand(const char *name,
                                 lldb::SBCommandPluginInterface *impl,
                                 const char *help, const char *syntax) {
  LLDB_RECORD_METHOD(lldb::SBCommand, SBCommandInterpreter, AddCommand,
                     (const char *, lldb::SBCommandPluginInterface *,
                      const char *, const char *),
                     name, impl, help, syntax);
  return LLDB_RECORD_RESULT(
      AddCommand(name, impl, help, syntax, /*auto_repeat_command=*/""))
}

lldb::SBCommand SBCommandInterpreter::AddCommand(
    const char *name, lldb::SBCommandPluginInterface *impl, const char *help,
    const char *syntax, const char *auto_repeat_command) {
  LLDB_RECORD_METHOD(lldb::SBCommand, SBCommandInterpreter, AddCommand,
                     (const char *, lldb::SBCommandPluginInterface *,
                      const char *, const char *, const char *),
                     name, impl, help, syntax, auto_repeat_command);

  lldb::CommandObjectSP new_command_sp;
  new_command_sp = std::make_shared<CommandPluginInterfaceImplementation>(
      *m_opaque_ptr, name, impl, help, syntax, /*flags=*/0,
      auto_repeat_command);

  if (new_command_sp &&
      m_opaque_ptr->AddUserCommand(name, new_command_sp, true))
    return LLDB_RECORD_RESULT(lldb::SBCommand(new_command_sp));
  return LLDB_RECORD_RESULT(lldb::SBCommand());
}

SBCommand::SBCommand() { LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBCommand); }

SBCommand::SBCommand(lldb::CommandObjectSP cmd_sp) : m_opaque_sp(cmd_sp) {}

bool SBCommand::IsValid() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBCommand, IsValid);
  return this->operator bool();
}
SBCommand::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBCommand, operator bool);

  return m_opaque_sp.get() != nullptr;
}

const char *SBCommand::GetName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBCommand, GetName);

  return (IsValid() ? ConstString(m_opaque_sp->GetCommandName()).AsCString() : nullptr);
}

const char *SBCommand::GetHelp() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBCommand, GetHelp);

  return (IsValid() ? ConstString(m_opaque_sp->GetHelp()).AsCString()
                    : nullptr);
}

const char *SBCommand::GetHelpLong() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBCommand, GetHelpLong);

  return (IsValid() ? ConstString(m_opaque_sp->GetHelpLong()).AsCString()
                    : nullptr);
}

void SBCommand::SetHelp(const char *help) {
  LLDB_RECORD_METHOD(void, SBCommand, SetHelp, (const char *), help);

  if (IsValid())
    m_opaque_sp->SetHelp(help);
}

void SBCommand::SetHelpLong(const char *help) {
  LLDB_RECORD_METHOD(void, SBCommand, SetHelpLong, (const char *), help);

  if (IsValid())
    m_opaque_sp->SetHelpLong(help);
}

lldb::SBCommand SBCommand::AddMultiwordCommand(const char *name,
                                               const char *help) {
  LLDB_RECORD_METHOD(lldb::SBCommand, SBCommand, AddMultiwordCommand,
                     (const char *, const char *), name, help);

  if (!IsValid())
    return LLDB_RECORD_RESULT(lldb::SBCommand());
  if (!m_opaque_sp->IsMultiwordObject())
    return LLDB_RECORD_RESULT(lldb::SBCommand());
  CommandObjectMultiword *new_command = new CommandObjectMultiword(
      m_opaque_sp->GetCommandInterpreter(), name, help);
  new_command->SetRemovable(true);
  lldb::CommandObjectSP new_command_sp(new_command);
  if (new_command_sp && m_opaque_sp->LoadSubCommand(name, new_command_sp))
    return LLDB_RECORD_RESULT(lldb::SBCommand(new_command_sp));
  return LLDB_RECORD_RESULT(lldb::SBCommand());
}

lldb::SBCommand SBCommand::AddCommand(const char *name,
                                      lldb::SBCommandPluginInterface *impl,
                                      const char *help) {
  LLDB_RECORD_METHOD(
      lldb::SBCommand, SBCommand, AddCommand,
      (const char *, lldb::SBCommandPluginInterface *, const char *), name,
      impl, help);
  return LLDB_RECORD_RESULT(AddCommand(name, impl, help, /*syntax=*/nullptr,
                                       /*auto_repeat_command=*/""))
}

lldb::SBCommand SBCommand::AddCommand(const char *name,
                                      lldb::SBCommandPluginInterface *impl,
                                      const char *help, const char *syntax) {
  LLDB_RECORD_METHOD(lldb::SBCommand, SBCommand, AddCommand,
                     (const char *, lldb::SBCommandPluginInterface *,
                      const char *, const char *),
                     name, impl, help, syntax);
  return LLDB_RECORD_RESULT(
      AddCommand(name, impl, help, syntax, /*auto_repeat_command=*/""))
}

lldb::SBCommand SBCommand::AddCommand(const char *name,
                                      lldb::SBCommandPluginInterface *impl,
                                      const char *help, const char *syntax,
                                      const char *auto_repeat_command) {
  LLDB_RECORD_METHOD(lldb::SBCommand, SBCommand, AddCommand,
                     (const char *, lldb::SBCommandPluginInterface *,
                      const char *, const char *, const char *),
                     name, impl, help, syntax, auto_repeat_command);

  if (!IsValid())
    return LLDB_RECORD_RESULT(lldb::SBCommand());
  if (!m_opaque_sp->IsMultiwordObject())
    return LLDB_RECORD_RESULT(lldb::SBCommand());
  lldb::CommandObjectSP new_command_sp;
  new_command_sp = std::make_shared<CommandPluginInterfaceImplementation>(
      m_opaque_sp->GetCommandInterpreter(), name, impl, help, syntax,
      /*flags=*/0, auto_repeat_command);
  if (new_command_sp && m_opaque_sp->LoadSubCommand(name, new_command_sp))
    return LLDB_RECORD_RESULT(lldb::SBCommand(new_command_sp));
  return LLDB_RECORD_RESULT(lldb::SBCommand());
}

uint32_t SBCommand::GetFlags() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBCommand, GetFlags);

  return (IsValid() ? m_opaque_sp->GetFlags().Get() : 0);
}

void SBCommand::SetFlags(uint32_t flags) {
  LLDB_RECORD_METHOD(void, SBCommand, SetFlags, (uint32_t), flags);

  if (IsValid())
    m_opaque_sp->GetFlags().Set(flags);
}

namespace lldb_private {
namespace repro {

template <> void RegisterMethods<SBCommandInterpreter>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBCommandInterpreter,
                            (lldb_private::CommandInterpreter *));
  LLDB_REGISTER_CONSTRUCTOR(SBCommandInterpreter,
                            (const lldb::SBCommandInterpreter &));
  LLDB_REGISTER_METHOD(
      const lldb::SBCommandInterpreter &,
      SBCommandInterpreter, operator=,(const lldb::SBCommandInterpreter &));
  LLDB_REGISTER_METHOD_CONST(bool, SBCommandInterpreter, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBCommandInterpreter, operator bool, ());
  LLDB_REGISTER_METHOD(bool, SBCommandInterpreter, CommandExists,
                       (const char *));
  LLDB_REGISTER_METHOD(bool, SBCommandInterpreter, AliasExists,
                       (const char *));
  LLDB_REGISTER_METHOD(bool, SBCommandInterpreter, IsActive, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBCommandInterpreter, WasInterrupted, ());
  LLDB_REGISTER_METHOD(const char *, SBCommandInterpreter,
                       GetIOHandlerControlSequence, (char));
  LLDB_REGISTER_METHOD(lldb::ReturnStatus, SBCommandInterpreter,
                       HandleCommand,
                       (const char *, lldb::SBCommandReturnObject &, bool));
  LLDB_REGISTER_METHOD(lldb::ReturnStatus, SBCommandInterpreter,
                       HandleCommand,
                       (const char *, lldb::SBExecutionContext &,
                        lldb::SBCommandReturnObject &, bool));
  LLDB_REGISTER_METHOD(void, SBCommandInterpreter, HandleCommandsFromFile,
                       (lldb::SBFileSpec &, lldb::SBExecutionContext &,
                        lldb::SBCommandInterpreterRunOptions &,
                        lldb::SBCommandReturnObject));
  LLDB_REGISTER_METHOD(int, SBCommandInterpreter, HandleCompletion,
                       (const char *, const char *, const char *, int, int,
                        lldb::SBStringList &));
  LLDB_REGISTER_METHOD(int, SBCommandInterpreter,
                       HandleCompletionWithDescriptions,
                       (const char *, const char *, const char *, int, int,
                        lldb::SBStringList &, lldb::SBStringList &));
  LLDB_REGISTER_METHOD(int, SBCommandInterpreter,
                       HandleCompletionWithDescriptions,
                       (const char *, uint32_t, int, int,
                        lldb::SBStringList &, lldb::SBStringList &));
  LLDB_REGISTER_METHOD(
      int, SBCommandInterpreter, HandleCompletion,
      (const char *, uint32_t, int, int, lldb::SBStringList &));
  LLDB_REGISTER_METHOD(bool, SBCommandInterpreter, HasCommands, ());
  LLDB_REGISTER_METHOD(bool, SBCommandInterpreter, HasAliases, ());
  LLDB_REGISTER_METHOD(bool, SBCommandInterpreter, HasAliasOptions, ());
  LLDB_REGISTER_METHOD(lldb::SBProcess, SBCommandInterpreter, GetProcess, ());
  LLDB_REGISTER_METHOD(lldb::SBDebugger, SBCommandInterpreter, GetDebugger,
                       ());
  LLDB_REGISTER_METHOD(bool, SBCommandInterpreter, GetPromptOnQuit, ());
  LLDB_REGISTER_METHOD(void, SBCommandInterpreter, SetPromptOnQuit, (bool));
  LLDB_REGISTER_METHOD(void, SBCommandInterpreter, AllowExitCodeOnQuit,
                       (bool));
  LLDB_REGISTER_METHOD(bool, SBCommandInterpreter, HasCustomQuitExitCode, ());
  LLDB_REGISTER_METHOD(int, SBCommandInterpreter, GetQuitStatus, ());
  LLDB_REGISTER_METHOD(void, SBCommandInterpreter, ResolveCommand,
                       (const char *, lldb::SBCommandReturnObject &));
  LLDB_REGISTER_METHOD(void, SBCommandInterpreter,
                       SourceInitFileInHomeDirectory,
                       (lldb::SBCommandReturnObject &));
  LLDB_REGISTER_METHOD(void, SBCommandInterpreter,
                       SourceInitFileInCurrentWorkingDirectory,
                       (lldb::SBCommandReturnObject &));
  LLDB_REGISTER_METHOD(lldb::SBBroadcaster, SBCommandInterpreter,
                       GetBroadcaster, ());
  LLDB_REGISTER_STATIC_METHOD(const char *, SBCommandInterpreter,
                              GetBroadcasterClass, ());
  LLDB_REGISTER_STATIC_METHOD(const char *, SBCommandInterpreter,
                              GetArgumentTypeAsCString,
                              (const lldb::CommandArgumentType));
  LLDB_REGISTER_STATIC_METHOD(const char *, SBCommandInterpreter,
                              GetArgumentDescriptionAsCString,
                              (const lldb::CommandArgumentType));
  LLDB_REGISTER_STATIC_METHOD(bool, SBCommandInterpreter,
                              EventIsCommandInterpreterEvent,
                              (const lldb::SBEvent &));
  LLDB_REGISTER_METHOD(lldb::SBCommand, SBCommandInterpreter,
                       AddMultiwordCommand, (const char *, const char *));
  LLDB_REGISTER_METHOD(
      lldb::SBCommand, SBCommandInterpreter, AddCommand,
      (const char *, lldb::SBCommandPluginInterface *, const char *));
  LLDB_REGISTER_METHOD(lldb::SBCommand, SBCommandInterpreter, AddCommand,
                       (const char *, lldb::SBCommandPluginInterface *,
                        const char *, const char *));
  LLDB_REGISTER_METHOD(lldb::SBCommand, SBCommandInterpreter, AddCommand,
                       (const char *, lldb::SBCommandPluginInterface *,
                        const char *, const char *, const char *));
  LLDB_REGISTER_CONSTRUCTOR(SBCommand, ());
  LLDB_REGISTER_METHOD(bool, SBCommand, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBCommand, operator bool, ());
  LLDB_REGISTER_METHOD(const char *, SBCommand, GetName, ());
  LLDB_REGISTER_METHOD(const char *, SBCommand, GetHelp, ());
  LLDB_REGISTER_METHOD(const char *, SBCommand, GetHelpLong, ());
  LLDB_REGISTER_METHOD(void, SBCommand, SetHelp, (const char *));
  LLDB_REGISTER_METHOD(void, SBCommand, SetHelpLong, (const char *));
  LLDB_REGISTER_METHOD(lldb::SBCommand, SBCommand, AddMultiwordCommand,
                       (const char *, const char *));
  LLDB_REGISTER_METHOD(
      lldb::SBCommand, SBCommand, AddCommand,
      (const char *, lldb::SBCommandPluginInterface *, const char *));
  LLDB_REGISTER_METHOD(lldb::SBCommand, SBCommand, AddCommand,
                       (const char *, lldb::SBCommandPluginInterface *,
                        const char *, const char *));
  LLDB_REGISTER_METHOD(lldb::SBCommand, SBCommand, AddCommand,
                       (const char *, lldb::SBCommandPluginInterface *,
                        const char *, const char *, const char *));
  LLDB_REGISTER_METHOD(uint32_t, SBCommand, GetFlags, ());
  LLDB_REGISTER_METHOD(void, SBCommand, SetFlags, (uint32_t));
}
}
}
