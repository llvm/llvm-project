#include "CommandObjectSession.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

class CommandObjectSessionSave : public CommandObjectParsed {
public:
  CommandObjectSessionSave(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "session save",
                            "Save the current session transcripts to a file.\n"
                            "If no file if specified, transcripts will be "
                            "saved to a temporary file.",
                            "session save [file]") {
    CommandArgumentEntry arg1;
    arg1.emplace_back(eArgTypePath, eArgRepeatOptional);
    m_arguments.push_back(arg1);
  }

  ~CommandObjectSessionSave() override = default;

  void
  HandleArgumentCompletion(CompletionRequest &request,
                           OptionElementVector &opt_element_vector) override {
    CommandCompletions::InvokeCommonCompletionCallbacks(
        GetCommandInterpreter(), CommandCompletions::eDiskFileCompletion,
        request, nullptr);
  }

protected:
  bool DoExecute(Args &args, CommandReturnObject &result) override {
    llvm::StringRef file_path;

    if (!args.empty())
      file_path = args[0].ref();

    if (m_interpreter.SaveTranscript(result, file_path.str()))
      result.SetStatus(eReturnStatusSuccessFinishNoResult);
    else
      result.SetStatus(eReturnStatusFailed);
    return result.Succeeded();
  }
};

CommandObjectSession::CommandObjectSession(CommandInterpreter &interpreter)
    : CommandObjectMultiword(interpreter, "session",
                             "Commands controlling LLDB session.",
                             "session <subcommand> [<command-options>]") {
  LoadSubCommand("save",
                 CommandObjectSP(new CommandObjectSessionSave(interpreter)));
  //  TODO: Move 'history' subcommand from CommandObjectCommands.
}
