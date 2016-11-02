//===-- BreakpointOptions.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointOptions.h"

#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StringList.h"
#include "lldb/Core/Value.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadSpec.h"

#include "llvm/ADT/STLExtras.h"

using namespace lldb;
using namespace lldb_private;

const char
    *BreakpointOptions::CommandData::g_option_names[static_cast<uint32_t>(
        BreakpointOptions::CommandData::OptionNames::LastOptionName)]{
        "UserSource", "ScriptSource", "StopOnError"};

StructuredData::ObjectSP
BreakpointOptions::CommandData::SerializeToStructuredData() {
  size_t num_strings = user_source.GetSize();
  if (num_strings == 0 && script_source.empty()) {
    // We shouldn't serialize commands if there aren't any, return an empty sp
    // to indicate this.
    return StructuredData::ObjectSP();
  }

  StructuredData::DictionarySP options_dict_sp(
      new StructuredData::Dictionary());
  options_dict_sp->AddBooleanItem(GetKey(OptionNames::StopOnError),
                                  stop_on_error);

  StructuredData::ArraySP user_source_sp(new StructuredData::Array());
  for (size_t i = 0; i < num_strings; i++) {
    StructuredData::StringSP item_sp(
        new StructuredData::String(user_source[i]));
    user_source_sp->AddItem(item_sp);
    options_dict_sp->AddItem(GetKey(OptionNames::UserSource), user_source_sp);
  }

  options_dict_sp->AddStringItem(
      GetKey(OptionNames::Interpreter),
      ScriptInterpreter::LanguageToString(interpreter));
  return options_dict_sp;
}

std::unique_ptr<BreakpointOptions::CommandData>
BreakpointOptions::CommandData::CreateFromStructuredData(
    const StructuredData::Dictionary &options_dict, Error &error) {
  std::unique_ptr<CommandData> data_up(new CommandData());
  bool found_something = false;

  bool success = options_dict.GetValueForKeyAsBoolean(
      GetKey(OptionNames::StopOnError), data_up->stop_on_error);

  if (success)
    found_something = true;

  std::string interpreter_str;
  ScriptLanguage interp_language;
  success = options_dict.GetValueForKeyAsString(
      GetKey(OptionNames::Interpreter), interpreter_str);

  if (!success) {
    error.SetErrorString("Missing command language value.");
    return data_up;
  }

  found_something = true;
  interp_language = ScriptInterpreter::StringToLanguage(interpreter_str);
  if (interp_language == eScriptLanguageUnknown) {
    error.SetErrorStringWithFormat("Unknown breakpoint command language: %s.",
                                   interpreter_str.c_str());
    return data_up;
  }
  data_up->interpreter = interp_language;

  StructuredData::Array *user_source;
  success = options_dict.GetValueForKeyAsArray(GetKey(OptionNames::UserSource),
                                               user_source);
  if (success) {
    found_something = true;
    size_t num_elems = user_source->GetSize();
    for (size_t i = 0; i < num_elems; i++) {
      std::string elem_string;
      success = user_source->GetItemAtIndexAsString(i, elem_string);
      if (success)
        data_up->user_source.AppendString(elem_string);
    }
  }

  if (found_something)
    return data_up;
  else
    return std::unique_ptr<BreakpointOptions::CommandData>();
}

const char *BreakpointOptions::g_option_names[(
    size_t)BreakpointOptions::OptionNames::LastOptionName]{
    "ConditionText", "IgnoreCount", "EnabledState", "OneShotState"};

bool BreakpointOptions::NullCallback(void *baton,
                                     StoppointCallbackContext *context,
                                     lldb::user_id_t break_id,
                                     lldb::user_id_t break_loc_id) {
  return true;
}

//----------------------------------------------------------------------
// BreakpointOptions constructor
//----------------------------------------------------------------------
BreakpointOptions::BreakpointOptions()
    : m_callback(BreakpointOptions::NullCallback), m_callback_baton_sp(),
      m_baton_is_command_baton(false), m_callback_is_synchronous(false),
      m_enabled(true), m_one_shot(false), m_ignore_count(0), m_thread_spec_ap(),
      m_condition_text(), m_condition_text_hash(0) {}

BreakpointOptions::BreakpointOptions(const char *condition, bool enabled,
                                     int32_t ignore, bool one_shot)
    : m_callback(nullptr), m_baton_is_command_baton(false),
      m_callback_is_synchronous(false), m_enabled(enabled),
      m_one_shot(one_shot), m_ignore_count(ignore), m_condition_text(condition),
      m_condition_text_hash(0)

{}

//----------------------------------------------------------------------
// BreakpointOptions copy constructor
//----------------------------------------------------------------------
BreakpointOptions::BreakpointOptions(const BreakpointOptions &rhs)
    : m_callback(rhs.m_callback), m_callback_baton_sp(rhs.m_callback_baton_sp),
      m_baton_is_command_baton(rhs.m_baton_is_command_baton),
      m_callback_is_synchronous(rhs.m_callback_is_synchronous),
      m_enabled(rhs.m_enabled), m_one_shot(rhs.m_one_shot),
      m_ignore_count(rhs.m_ignore_count), m_thread_spec_ap() {
  if (rhs.m_thread_spec_ap.get() != nullptr)
    m_thread_spec_ap.reset(new ThreadSpec(*rhs.m_thread_spec_ap.get()));
  m_condition_text = rhs.m_condition_text;
  m_condition_text_hash = rhs.m_condition_text_hash;
}

//----------------------------------------------------------------------
// BreakpointOptions assignment operator
//----------------------------------------------------------------------
const BreakpointOptions &BreakpointOptions::
operator=(const BreakpointOptions &rhs) {
  m_callback = rhs.m_callback;
  m_callback_baton_sp = rhs.m_callback_baton_sp;
  m_baton_is_command_baton = rhs.m_baton_is_command_baton;
  m_callback_is_synchronous = rhs.m_callback_is_synchronous;
  m_enabled = rhs.m_enabled;
  m_one_shot = rhs.m_one_shot;
  m_ignore_count = rhs.m_ignore_count;
  if (rhs.m_thread_spec_ap.get() != nullptr)
    m_thread_spec_ap.reset(new ThreadSpec(*rhs.m_thread_spec_ap.get()));
  m_condition_text = rhs.m_condition_text;
  m_condition_text_hash = rhs.m_condition_text_hash;
  return *this;
}

BreakpointOptions *
BreakpointOptions::CopyOptionsNoCallback(BreakpointOptions &orig) {
  BreakpointHitCallback orig_callback = orig.m_callback;
  lldb::BatonSP orig_callback_baton_sp = orig.m_callback_baton_sp;
  bool orig_is_sync = orig.m_callback_is_synchronous;

  orig.ClearCallback();
  BreakpointOptions *ret_val = new BreakpointOptions(orig);

  orig.SetCallback(orig_callback, orig_callback_baton_sp, orig_is_sync);

  return ret_val;
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
BreakpointOptions::~BreakpointOptions() = default;

std::unique_ptr<BreakpointOptions> BreakpointOptions::CreateFromStructuredData(
    Target &target, const StructuredData::Dictionary &options_dict,
    Error &error) {
  bool enabled = true;
  bool one_shot = false;
  int32_t ignore_count = 0;
  std::string condition_text;

  bool success = options_dict.GetValueForKeyAsBoolean(
      GetKey(OptionNames::EnabledState), enabled);
  if (!success) {
    error.SetErrorStringWithFormat("%s key is not a boolean.",
                                   GetKey(OptionNames::EnabledState));
    return nullptr;
  }

  success = options_dict.GetValueForKeyAsBoolean(
      GetKey(OptionNames::OneShotState), one_shot);
  if (!success) {
    error.SetErrorStringWithFormat("%s key is not a boolean.",
                                   GetKey(OptionNames::OneShotState));
    return nullptr;
  }
  success = options_dict.GetValueForKeyAsInteger(
      GetKey(OptionNames::IgnoreCount), ignore_count);
  if (!success) {
    error.SetErrorStringWithFormat("%s key is not an integer.",
                                   GetKey(OptionNames::IgnoreCount));
    return nullptr;
  }

  std::unique_ptr<CommandData> cmd_data_up;
  StructuredData::Dictionary *cmds_dict;
  success = options_dict.GetValueForKeyAsDictionary(
      CommandData::GetSerializationKey(), cmds_dict);
  if (success && cmds_dict) {
    Error cmds_error;
    cmd_data_up = CommandData::CreateFromStructuredData(*cmds_dict, cmds_error);
    if (cmds_error.Fail()) {
      error.SetErrorStringWithFormat(
          "Failed to deserialize breakpoint command options: %s.",
          cmds_error.AsCString());
      return nullptr;
    }
  }

  auto bp_options = llvm::make_unique<BreakpointOptions>(
      condition_text.c_str(), enabled, ignore_count, one_shot);
  if (cmd_data_up.get()) {
    if (cmd_data_up->interpreter == eScriptLanguageNone)
      bp_options->SetCommandDataCallback(cmd_data_up);
    else {
      ScriptInterpreter *interp =
          target.GetDebugger().GetCommandInterpreter().GetScriptInterpreter();
      if (!interp) {
        error.SetErrorStringWithFormat(
            "Can't set script commands - no script interpreter");
        return nullptr;
      }
      if (interp->GetLanguage() != cmd_data_up->interpreter) {
        error.SetErrorStringWithFormat(
            "Current script language doesn't match breakpoint's language: %s",
            ScriptInterpreter::LanguageToString(cmd_data_up->interpreter)
                .c_str());
        return nullptr;
      }
      Error script_error;
      script_error =
          interp->SetBreakpointCommandCallback(bp_options.get(), cmd_data_up);
      if (script_error.Fail()) {
        error.SetErrorStringWithFormat("Error generating script callback: %s.",
                                       error.AsCString());
        return nullptr;
      }
    }
  }

  StructuredData::Dictionary *thread_spec_dict;
  success = options_dict.GetValueForKeyAsDictionary(
      ThreadSpec::GetSerializationKey(), thread_spec_dict);
  if (success) {
    Error thread_spec_error;
    std::unique_ptr<ThreadSpec> thread_spec_up =
        ThreadSpec::CreateFromStructuredData(*thread_spec_dict,
                                             thread_spec_error);
    if (thread_spec_error.Fail()) {
      error.SetErrorStringWithFormat(
          "Failed to deserialize breakpoint thread spec options: %s.",
          thread_spec_error.AsCString());
      return nullptr;
    }
    bp_options->SetThreadSpec(thread_spec_up);
  }
  return bp_options;
}

StructuredData::ObjectSP BreakpointOptions::SerializeToStructuredData() {
  StructuredData::DictionarySP options_dict_sp(
      new StructuredData::Dictionary());
  options_dict_sp->AddBooleanItem(GetKey(OptionNames::EnabledState), m_enabled);
  options_dict_sp->AddBooleanItem(GetKey(OptionNames::OneShotState),
                                  m_one_shot);
  options_dict_sp->AddIntegerItem(GetKey(OptionNames::IgnoreCount),
                                  m_ignore_count);
  options_dict_sp->AddStringItem(GetKey(OptionNames::ConditionText),
                                 m_condition_text);
  if (m_baton_is_command_baton) {
    auto cmd_baton =
        std::static_pointer_cast<CommandBaton>(m_callback_baton_sp);
    StructuredData::ObjectSP commands_sp =
        cmd_baton->getItem()->SerializeToStructuredData();
    if (commands_sp) {
      options_dict_sp->AddItem(
          BreakpointOptions::CommandData::GetSerializationKey(), commands_sp);
    }
  }
  if (m_thread_spec_ap) {
    StructuredData::ObjectSP thread_spec_sp =
        m_thread_spec_ap->SerializeToStructuredData();
    options_dict_sp->AddItem(ThreadSpec::GetSerializationKey(), thread_spec_sp);
  }

  return options_dict_sp;
}

//------------------------------------------------------------------
// Callbacks
//------------------------------------------------------------------
void BreakpointOptions::SetCallback(BreakpointHitCallback callback,
                                    const lldb::BatonSP &callback_baton_sp,
                                    bool callback_is_synchronous) {
  // FIXME: This seems unsafe.  If BatonSP actually *is* a CommandBaton, but
  // in a shared_ptr<Baton> instead of a shared_ptr<CommandBaton>, then we
  // will set m_baton_is_command_baton to false, which is incorrect.
  // One possible solution is to make the base Baton class provide a method
  // such as:
  //     virtual StringRef getBatonId() const { return ""; }
  // and have CommandBaton override this to return something unique, and then
  // check for it here.  Another option might be to make Baton using the llvm
  // casting infrastructure, so that we could write something like:
  //     if (llvm::isa<CommandBaton>(callback_baton_sp))
  // at relevant callsites instead of storing a boolean.
  m_callback_is_synchronous = callback_is_synchronous;
  m_callback = callback;
  m_callback_baton_sp = callback_baton_sp;
  m_baton_is_command_baton = false;
}

void BreakpointOptions::SetCallback(
    BreakpointHitCallback callback,
    const BreakpointOptions::CommandBatonSP &callback_baton_sp,
    bool callback_is_synchronous) {
  m_callback_is_synchronous = callback_is_synchronous;
  m_callback = callback;
  m_callback_baton_sp = callback_baton_sp;
  m_baton_is_command_baton = true;
}

void BreakpointOptions::ClearCallback() {
  m_callback = BreakpointOptions::NullCallback;
  m_callback_is_synchronous = false;
  m_callback_baton_sp.reset();
  m_baton_is_command_baton = false;
}

Baton *BreakpointOptions::GetBaton() { return m_callback_baton_sp.get(); }

const Baton *BreakpointOptions::GetBaton() const {
  return m_callback_baton_sp.get();
}

bool BreakpointOptions::InvokeCallback(StoppointCallbackContext *context,
                                       lldb::user_id_t break_id,
                                       lldb::user_id_t break_loc_id) {
  if (m_callback && context->is_synchronous == IsCallbackSynchronous()) {
    return m_callback(m_callback_baton_sp ? m_callback_baton_sp->data()
                                          : nullptr,
                      context, break_id, break_loc_id);
  } else
    return true;
}

bool BreakpointOptions::HasCallback() const {
  return m_callback != BreakpointOptions::NullCallback;
}

bool BreakpointOptions::GetCommandLineCallbacks(StringList &command_list) {
  if (!HasCallback())
    return false;
  if (!m_baton_is_command_baton)
    return false;

  auto cmd_baton = std::static_pointer_cast<CommandBaton>(m_callback_baton_sp);
  CommandData *data = cmd_baton->getItem();
  if (!data)
    return false;
  command_list = data->user_source;
  return true;
}

void BreakpointOptions::SetCondition(const char *condition) {
  if (!condition)
    condition = "";

  m_condition_text.assign(condition);
  std::hash<std::string> hasher;
  m_condition_text_hash = hasher(m_condition_text);
}

const char *BreakpointOptions::GetConditionText(size_t *hash) const {
  if (!m_condition_text.empty()) {
    if (hash)
      *hash = m_condition_text_hash;

    return m_condition_text.c_str();
  } else {
    return nullptr;
  }
}

const ThreadSpec *BreakpointOptions::GetThreadSpecNoCreate() const {
  return m_thread_spec_ap.get();
}

ThreadSpec *BreakpointOptions::GetThreadSpec() {
  if (m_thread_spec_ap.get() == nullptr)
    m_thread_spec_ap.reset(new ThreadSpec());

  return m_thread_spec_ap.get();
}

void BreakpointOptions::SetThreadID(lldb::tid_t thread_id) {
  GetThreadSpec()->SetTID(thread_id);
}

void BreakpointOptions::SetThreadSpec(
    std::unique_ptr<ThreadSpec> &thread_spec_up) {
  m_thread_spec_ap = std::move(thread_spec_up);
}

void BreakpointOptions::GetDescription(Stream *s,
                                       lldb::DescriptionLevel level) const {
  // Figure out if there are any options not at their default value, and only
  // print
  // anything if there are:

  if (m_ignore_count != 0 || !m_enabled || m_one_shot ||
      (GetThreadSpecNoCreate() != nullptr &&
       GetThreadSpecNoCreate()->HasSpecification())) {
    if (level == lldb::eDescriptionLevelVerbose) {
      s->EOL();
      s->IndentMore();
      s->Indent();
      s->PutCString("Breakpoint Options:\n");
      s->IndentMore();
      s->Indent();
    } else
      s->PutCString(" Options: ");

    if (m_ignore_count > 0)
      s->Printf("ignore: %d ", m_ignore_count);
    s->Printf("%sabled ", m_enabled ? "en" : "dis");

    if (m_one_shot)
      s->Printf("one-shot ");

    if (m_thread_spec_ap.get())
      m_thread_spec_ap->GetDescription(s, level);

    if (level == lldb::eDescriptionLevelFull) {
      s->IndentLess();
      s->IndentMore();
    }
  }

  if (m_callback_baton_sp.get()) {
    if (level != eDescriptionLevelBrief) {
      s->EOL();
      m_callback_baton_sp->GetDescription(s, level);
    }
  }
  if (!m_condition_text.empty()) {
    if (level != eDescriptionLevelBrief) {
      s->EOL();
      s->Printf("Condition: %s\n", m_condition_text.c_str());
    }
  }
}

void BreakpointOptions::CommandBaton::GetDescription(
    Stream *s, lldb::DescriptionLevel level) const {
  const CommandData *data = getItem();

  if (level == eDescriptionLevelBrief) {
    s->Printf(", commands = %s",
              (data && data->user_source.GetSize() > 0) ? "yes" : "no");
    return;
  }

  s->IndentMore();
  s->Indent("Breakpoint commands");
  if (data->interpreter != eScriptLanguageNone)
    s->Printf(" (%s):\n",
              ScriptInterpreter::LanguageToString(data->interpreter).c_str());
  else
    s->PutCString(":\n");

  s->IndentMore();
  if (data && data->user_source.GetSize() > 0) {
    const size_t num_strings = data->user_source.GetSize();
    for (size_t i = 0; i < num_strings; ++i) {
      s->Indent(data->user_source.GetStringAtIndex(i));
      s->EOL();
    }
  } else {
    s->PutCString("No commands.\n");
  }
  s->IndentLess();
  s->IndentLess();
}

void BreakpointOptions::SetCommandDataCallback(
    std::unique_ptr<CommandData> &cmd_data) {
  cmd_data->interpreter = eScriptLanguageNone;
  auto baton_sp = std::make_shared<CommandBaton>(std::move(cmd_data));
  SetCallback(BreakpointOptions::BreakpointOptionsCallbackFunction, baton_sp);
}

bool BreakpointOptions::BreakpointOptionsCallbackFunction(
    void *baton, StoppointCallbackContext *context, lldb::user_id_t break_id,
    lldb::user_id_t break_loc_id) {
  bool ret_value = true;
  if (baton == nullptr)
    return true;

  CommandData *data = (CommandData *)baton;
  StringList &commands = data->user_source;

  if (commands.GetSize() > 0) {
    ExecutionContext exe_ctx(context->exe_ctx_ref);
    Target *target = exe_ctx.GetTargetPtr();
    if (target) {
      CommandReturnObject result;
      Debugger &debugger = target->GetDebugger();
      // Rig up the results secondary output stream to the debugger's, so the
      // output will come out synchronously
      // if the debugger is set up that way.

      StreamSP output_stream(debugger.GetAsyncOutputStream());
      StreamSP error_stream(debugger.GetAsyncErrorStream());
      result.SetImmediateOutputStream(output_stream);
      result.SetImmediateErrorStream(error_stream);

      CommandInterpreterRunOptions options;
      options.SetStopOnContinue(true);
      options.SetStopOnError(data->stop_on_error);
      options.SetEchoCommands(true);
      options.SetPrintResults(true);
      options.SetAddToHistory(false);

      debugger.GetCommandInterpreter().HandleCommands(commands, &exe_ctx,
                                                      options, result);
      result.GetImmediateOutputStream()->Flush();
      result.GetImmediateErrorStream()->Flush();
    }
  }
  return ret_value;
}
