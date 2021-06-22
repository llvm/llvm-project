//===-- CommandObjectReproducer.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectReproducer.h"

#include "lldb/Host/HostInfo.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/OptionArgParser.h"
#include "lldb/Utility/GDBRemote.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/Reproducer.h"

#include <csignal>

using namespace lldb;
using namespace llvm;
using namespace lldb_private;
using namespace lldb_private::repro;

enum ReproducerProvider {
  eReproducerProviderCommands,
  eReproducerProviderFiles,
  eReproducerProviderSymbolFiles,
  eReproducerProviderGDB,
  eReproducerProviderProcessInfo,
  eReproducerProviderVersion,
  eReproducerProviderWorkingDirectory,
  eReproducerProviderHomeDirectory,
  eReproducerProviderNone
};

static constexpr OptionEnumValueElement g_reproducer_provider_type[] = {
    {
        eReproducerProviderCommands,
        "commands",
        "Command Interpreter Commands",
    },
    {
        eReproducerProviderFiles,
        "files",
        "Files",
    },
    {
        eReproducerProviderSymbolFiles,
        "symbol-files",
        "Symbol Files",
    },
    {
        eReproducerProviderGDB,
        "gdb",
        "GDB Remote Packets",
    },
    {
        eReproducerProviderProcessInfo,
        "processes",
        "Process Info",
    },
    {
        eReproducerProviderVersion,
        "version",
        "Version",
    },
    {
        eReproducerProviderWorkingDirectory,
        "cwd",
        "Working Directory",
    },
    {
        eReproducerProviderHomeDirectory,
        "home",
        "Home Directory",
    },
    {
        eReproducerProviderNone,
        "none",
        "None",
    },
};

static constexpr OptionEnumValues ReproducerProviderType() {
  return OptionEnumValues(g_reproducer_provider_type);
}

#define LLDB_OPTIONS_reproducer_dump
#include "CommandOptions.inc"

enum ReproducerCrashSignal {
  eReproducerCrashSigill,
  eReproducerCrashSigsegv,
};

static constexpr OptionEnumValueElement g_reproducer_signaltype[] = {
    {
        eReproducerCrashSigill,
        "SIGILL",
        "Illegal instruction",
    },
    {
        eReproducerCrashSigsegv,
        "SIGSEGV",
        "Segmentation fault",
    },
};

static constexpr OptionEnumValues ReproducerSignalType() {
  return OptionEnumValues(g_reproducer_signaltype);
}

#define LLDB_OPTIONS_reproducer_xcrash
#include "CommandOptions.inc"

#define LLDB_OPTIONS_reproducer_verify
#include "CommandOptions.inc"

template <typename T>
llvm::Expected<T> static ReadFromYAML(StringRef filename) {
  auto error_or_file = MemoryBuffer::getFile(filename);
  if (auto err = error_or_file.getError()) {
    return errorCodeToError(err);
  }

  T t;
  yaml::Input yin((*error_or_file)->getBuffer());
  yin >> t;

  if (auto err = yin.error()) {
    return errorCodeToError(err);
  }

  return t;
}

static void SetError(CommandReturnObject &result, Error err) {
  result.AppendError(toString(std::move(err)));
}

/// Create a loader from the given path if specified. Otherwise use the current
/// loader used for replay.
static Loader *
GetLoaderFromPathOrCurrent(llvm::Optional<Loader> &loader_storage,
                           CommandReturnObject &result,
                           FileSpec reproducer_path) {
  if (reproducer_path) {
    loader_storage.emplace(reproducer_path);
    Loader *loader = &(*loader_storage);
    if (Error err = loader->LoadIndex()) {
      // This is a hard error and will set the result to eReturnStatusFailed.
      SetError(result, std::move(err));
      return nullptr;
    }
    return loader;
  }

  if (Loader *loader = Reproducer::Instance().GetLoader())
    return loader;

  // This is a soft error because this is expected to fail during capture.
  result.AppendError(
      "Not specifying a reproducer is only support during replay.");
  result.SetStatus(eReturnStatusSuccessFinishNoResult);
  return nullptr;
}

class CommandObjectReproducerGenerate : public CommandObjectParsed {
public:
  CommandObjectReproducerGenerate(CommandInterpreter &interpreter)
      : CommandObjectParsed(
            interpreter, "reproducer generate",
            "Generate reproducer on disk. When the debugger is in capture "
            "mode, this command will output the reproducer to a directory on "
            "disk and quit. In replay mode this command in a no-op.",
            nullptr) {}

  ~CommandObjectReproducerGenerate() override = default;

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    if (!command.empty()) {
      result.AppendErrorWithFormat("'%s' takes no arguments",
                                   m_cmd_name.c_str());
      return false;
    }

    auto &r = Reproducer::Instance();
    if (auto generator = r.GetGenerator()) {
      generator->Keep();
      if (llvm::Error e = repro::Finalize(r.GetReproducerPath())) {
        SetError(result, std::move(e));
        return result.Succeeded();
      }
    } else if (r.IsReplaying()) {
      // Make this operation a NO-OP in replay mode.
      result.SetStatus(eReturnStatusSuccessFinishNoResult);
      return result.Succeeded();
    } else {
      result.AppendErrorWithFormat("Unable to get the reproducer generator");
      return false;
    }

    result.GetOutputStream()
        << "Reproducer written to '" << r.GetReproducerPath() << "'\n";
    result.GetOutputStream()
        << "Please have a look at the directory to assess if you're willing to "
           "share the contained information.\n";

    m_interpreter.BroadcastEvent(
        CommandInterpreter::eBroadcastBitQuitCommandReceived);
    result.SetStatus(eReturnStatusQuit);
    return result.Succeeded();
  }
};

class CommandObjectReproducerXCrash : public CommandObjectParsed {
public:
  CommandObjectReproducerXCrash(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "reproducer xcrash",
                            "Intentionally force  the debugger to crash in "
                            "order to trigger and test reproducer generation.",
                            nullptr) {}

  ~CommandObjectReproducerXCrash() override = default;

  Options *GetOptions() override { return &m_options; }

  class CommandOptions : public Options {
  public:
    CommandOptions() : Options() {}

    ~CommandOptions() override = default;

    Status SetOptionValue(uint32_t option_idx, StringRef option_arg,
                          ExecutionContext *execution_context) override {
      Status error;
      const int short_option = m_getopt_table[option_idx].val;

      switch (short_option) {
      case 's':
        signal = (ReproducerCrashSignal)OptionArgParser::ToOptionEnum(
            option_arg, GetDefinitions()[option_idx].enum_values, 0, error);
        if (!error.Success())
          error.SetErrorStringWithFormat("unrecognized value for signal '%s'",
                                         option_arg.str().c_str());
        break;
      default:
        llvm_unreachable("Unimplemented option");
      }

      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      signal = eReproducerCrashSigsegv;
    }

    ArrayRef<OptionDefinition> GetDefinitions() override {
      return makeArrayRef(g_reproducer_xcrash_options);
    }

    ReproducerCrashSignal signal = eReproducerCrashSigsegv;
  };

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    if (!command.empty()) {
      result.AppendErrorWithFormat("'%s' takes no arguments",
                                   m_cmd_name.c_str());
      return false;
    }

    auto &r = Reproducer::Instance();

    if (!r.IsCapturing() && !r.IsReplaying()) {
      result.AppendError(
          "forcing a crash is only supported when capturing a reproducer.");
      result.SetStatus(eReturnStatusSuccessFinishNoResult);
      return false;
    }

    switch (m_options.signal) {
    case eReproducerCrashSigill:
      std::raise(SIGILL);
      break;
    case eReproducerCrashSigsegv:
      std::raise(SIGSEGV);
      break;
    }

    result.SetStatus(eReturnStatusQuit);
    return result.Succeeded();
  }

private:
  CommandOptions m_options;
};

class CommandObjectReproducerStatus : public CommandObjectParsed {
public:
  CommandObjectReproducerStatus(CommandInterpreter &interpreter)
      : CommandObjectParsed(
            interpreter, "reproducer status",
            "Show the current reproducer status. In capture mode the "
            "debugger "
            "is collecting all the information it needs to create a "
            "reproducer.  In replay mode the reproducer is replaying a "
            "reproducer. When the reproducers are off, no data is collected "
            "and no reproducer can be generated.",
            nullptr) {}

  ~CommandObjectReproducerStatus() override = default;

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    if (!command.empty()) {
      result.AppendErrorWithFormat("'%s' takes no arguments",
                                   m_cmd_name.c_str());
      return false;
    }

    auto &r = Reproducer::Instance();
    if (r.IsCapturing()) {
      result.GetOutputStream() << "Reproducer is in capture mode.\n";
    } else if (r.IsReplaying()) {
      result.GetOutputStream() << "Reproducer is in replay mode.\n";
    } else {
      result.GetOutputStream() << "Reproducer is off.\n";
    }

    if (r.IsCapturing() || r.IsReplaying()) {
      result.GetOutputStream()
          << "Path: " << r.GetReproducerPath().GetPath() << '\n';
    }

    // Auto generate is hidden unless enabled because this is mostly for
    // development and testing.
    if (Generator *g = r.GetGenerator()) {
      if (g->IsAutoGenerate())
        result.GetOutputStream() << "Auto generate: on\n";
    }

    result.SetStatus(eReturnStatusSuccessFinishResult);
    return result.Succeeded();
  }
};

class CommandObjectReproducerDump : public CommandObjectParsed {
public:
  CommandObjectReproducerDump(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "reproducer dump",
                            "Dump the information contained in a reproducer. "
                            "If no reproducer is specified during replay, it "
                            "dumps the content of the current reproducer.",
                            nullptr) {}

  ~CommandObjectReproducerDump() override = default;

  Options *GetOptions() override { return &m_options; }

  class CommandOptions : public Options {
  public:
    CommandOptions() : Options(), file() {}

    ~CommandOptions() override = default;

    Status SetOptionValue(uint32_t option_idx, StringRef option_arg,
                          ExecutionContext *execution_context) override {
      Status error;
      const int short_option = m_getopt_table[option_idx].val;

      switch (short_option) {
      case 'f':
        file.SetFile(option_arg, FileSpec::Style::native);
        FileSystem::Instance().Resolve(file);
        break;
      case 'p':
        provider = (ReproducerProvider)OptionArgParser::ToOptionEnum(
            option_arg, GetDefinitions()[option_idx].enum_values, 0, error);
        if (!error.Success())
          error.SetErrorStringWithFormat("unrecognized value for provider '%s'",
                                         option_arg.str().c_str());
        break;
      default:
        llvm_unreachable("Unimplemented option");
      }

      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      file.Clear();
      provider = eReproducerProviderNone;
    }

    ArrayRef<OptionDefinition> GetDefinitions() override {
      return makeArrayRef(g_reproducer_dump_options);
    }

    FileSpec file;
    ReproducerProvider provider = eReproducerProviderNone;
  };

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    if (!command.empty()) {
      result.AppendErrorWithFormat("'%s' takes no arguments",
                                   m_cmd_name.c_str());
      return false;
    }

    llvm::Optional<Loader> loader_storage;
    Loader *loader =
        GetLoaderFromPathOrCurrent(loader_storage, result, m_options.file);
    if (!loader)
      return false;

    switch (m_options.provider) {
    case eReproducerProviderFiles: {
      FileSpec vfs_mapping = loader->GetFile<FileProvider::Info>();

      // Read the VFS mapping.
      ErrorOr<std::unique_ptr<MemoryBuffer>> buffer =
          vfs::getRealFileSystem()->getBufferForFile(vfs_mapping.GetPath());
      if (!buffer) {
        SetError(result, errorCodeToError(buffer.getError()));
        return false;
      }

      // Initialize a VFS from the given mapping.
      IntrusiveRefCntPtr<vfs::FileSystem> vfs = vfs::getVFSFromYAML(
          std::move(buffer.get()), nullptr, vfs_mapping.GetPath());

      // Dump the VFS to a buffer.
      std::string str;
      raw_string_ostream os(str);
      static_cast<vfs::RedirectingFileSystem &>(*vfs).dump(os);
      os.flush();

      // Return the string.
      result.AppendMessage(str);
      result.SetStatus(eReturnStatusSuccessFinishResult);
      return true;
    }
    case eReproducerProviderSymbolFiles: {
      Expected<std::string> symbol_files =
          loader->LoadBuffer<SymbolFileProvider>();
      if (!symbol_files) {
        SetError(result, symbol_files.takeError());
        return false;
      }

      std::vector<SymbolFileProvider::Entry> entries;
      llvm::yaml::Input yin(*symbol_files);
      yin >> entries;

      for (const auto &entry : entries) {
        result.AppendMessageWithFormat("- uuid:        %s\n",
                                       entry.uuid.c_str());
        result.AppendMessageWithFormat("  module path: %s\n",
                                       entry.module_path.c_str());
        result.AppendMessageWithFormat("  symbol path: %s\n",
                                       entry.symbol_path.c_str());
      }
      result.SetStatus(eReturnStatusSuccessFinishResult);
      return true;
    }
    case eReproducerProviderVersion: {
      Expected<std::string> version = loader->LoadBuffer<VersionProvider>();
      if (!version) {
        SetError(result, version.takeError());
        return false;
      }
      result.AppendMessage(*version);
      result.SetStatus(eReturnStatusSuccessFinishResult);
      return true;
    }
    case eReproducerProviderWorkingDirectory: {
      Expected<std::string> cwd =
          repro::GetDirectoryFrom<WorkingDirectoryProvider>(loader);
      if (!cwd) {
        SetError(result, cwd.takeError());
        return false;
      }
      result.AppendMessage(*cwd);
      result.SetStatus(eReturnStatusSuccessFinishResult);
      return true;
    }
    case eReproducerProviderHomeDirectory: {
      Expected<std::string> home =
          repro::GetDirectoryFrom<HomeDirectoryProvider>(loader);
      if (!home) {
        SetError(result, home.takeError());
        return false;
      }
      result.AppendMessage(*home);
      result.SetStatus(eReturnStatusSuccessFinishResult);
      return true;
    }
    case eReproducerProviderCommands: {
      std::unique_ptr<repro::MultiLoader<repro::CommandProvider>> multi_loader =
          repro::MultiLoader<repro::CommandProvider>::Create(loader);
      if (!multi_loader) {
        SetError(result,
                 make_error<StringError>("Unable to create command loader.",
                                         llvm::inconvertibleErrorCode()));
        return false;
      }

      // Iterate over the command files and dump them.
      llvm::Optional<std::string> command_file;
      while ((command_file = multi_loader->GetNextFile())) {
        if (!command_file)
          break;

        auto command_buffer = llvm::MemoryBuffer::getFile(*command_file);
        if (auto err = command_buffer.getError()) {
          SetError(result, errorCodeToError(err));
          return false;
        }
        result.AppendMessage((*command_buffer)->getBuffer());
      }

      result.SetStatus(eReturnStatusSuccessFinishResult);
      return true;
    }
    case eReproducerProviderGDB: {
      std::unique_ptr<repro::MultiLoader<repro::GDBRemoteProvider>>
          multi_loader =
              repro::MultiLoader<repro::GDBRemoteProvider>::Create(loader);

      if (!multi_loader) {
        SetError(result,
                 make_error<StringError>("Unable to create GDB loader.",
                                         llvm::inconvertibleErrorCode()));
        return false;
      }

      llvm::Optional<std::string> gdb_file;
      while ((gdb_file = multi_loader->GetNextFile())) {
        if (llvm::Expected<std::vector<GDBRemotePacket>> packets =
                ReadFromYAML<std::vector<GDBRemotePacket>>(*gdb_file)) {
          for (GDBRemotePacket &packet : *packets) {
            packet.Dump(result.GetOutputStream());
          }
        } else {
          SetError(result, packets.takeError());
          return false;
        }
      }

      result.SetStatus(eReturnStatusSuccessFinishResult);
      return true;
    }
    case eReproducerProviderProcessInfo: {
      std::unique_ptr<repro::MultiLoader<repro::ProcessInfoProvider>>
          multi_loader =
              repro::MultiLoader<repro::ProcessInfoProvider>::Create(loader);

      if (!multi_loader) {
        SetError(result, make_error<StringError>(
                             llvm::inconvertibleErrorCode(),
                             "Unable to create process info loader."));
        return false;
      }

      llvm::Optional<std::string> process_file;
      while ((process_file = multi_loader->GetNextFile())) {
        if (llvm::Expected<ProcessInstanceInfoList> infos =
                ReadFromYAML<ProcessInstanceInfoList>(*process_file)) {
          for (ProcessInstanceInfo info : *infos)
            info.Dump(result.GetOutputStream(), HostInfo::GetUserIDResolver());
        } else {
          SetError(result, infos.takeError());
          return false;
        }
      }

      result.SetStatus(eReturnStatusSuccessFinishResult);
      return true;
    }
    case eReproducerProviderNone:
      result.AppendError("No valid provider specified.");
      return false;
    }

    result.SetStatus(eReturnStatusSuccessFinishNoResult);
    return result.Succeeded();
  }

private:
  CommandOptions m_options;
};

class CommandObjectReproducerVerify : public CommandObjectParsed {
public:
  CommandObjectReproducerVerify(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "reproducer verify",
                            "Verify the contents of a reproducer. "
                            "If no reproducer is specified during replay, it "
                            "verifies the content of the current reproducer.",
                            nullptr) {}

  ~CommandObjectReproducerVerify() override = default;

  Options *GetOptions() override { return &m_options; }

  class CommandOptions : public Options {
  public:
    CommandOptions() : Options(), file() {}

    ~CommandOptions() override = default;

    Status SetOptionValue(uint32_t option_idx, StringRef option_arg,
                          ExecutionContext *execution_context) override {
      Status error;
      const int short_option = m_getopt_table[option_idx].val;

      switch (short_option) {
      case 'f':
        file.SetFile(option_arg, FileSpec::Style::native);
        FileSystem::Instance().Resolve(file);
        break;
      default:
        llvm_unreachable("Unimplemented option");
      }

      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      file.Clear();
    }

    ArrayRef<OptionDefinition> GetDefinitions() override {
      return makeArrayRef(g_reproducer_verify_options);
    }

    FileSpec file;
  };

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    if (!command.empty()) {
      result.AppendErrorWithFormat("'%s' takes no arguments",
                                   m_cmd_name.c_str());
      return false;
    }

    llvm::Optional<Loader> loader_storage;
    Loader *loader =
        GetLoaderFromPathOrCurrent(loader_storage, result, m_options.file);
    if (!loader)
      return false;

    bool errors = false;
    auto error_callback = [&](llvm::StringRef error) {
      errors = true;
      result.AppendError(error);
    };

    bool warnings = false;
    auto warning_callback = [&](llvm::StringRef warning) {
      warnings = true;
      result.AppendWarning(warning);
    };

    auto note_callback = [&](llvm::StringRef warning) {
      result.AppendMessage(warning);
    };

    Verifier verifier(loader);
    verifier.Verify(error_callback, warning_callback, note_callback);

    if (warnings || errors) {
      result.AppendMessage("reproducer verification failed");
      result.SetStatus(eReturnStatusFailed);
    } else {
      result.AppendMessage("reproducer verification succeeded");
      result.SetStatus(eReturnStatusSuccessFinishResult);
    }

    return result.Succeeded();
  }

private:
  CommandOptions m_options;
};

CommandObjectReproducer::CommandObjectReproducer(
    CommandInterpreter &interpreter)
    : CommandObjectMultiword(
          interpreter, "reproducer",
          "Commands for manipulating reproducers. Reproducers make it "
          "possible "
          "to capture full debug sessions with all its dependencies. The "
          "resulting reproducer is used to replay the debug session while "
          "debugging the debugger.\n"
          "Because reproducers need the whole the debug session from "
          "beginning to end, you need to launch the debugger in capture or "
          "replay mode, commonly though the command line driver.\n"
          "Reproducers are unrelated record-replay debugging, as you cannot "
          "interact with the debugger during replay.\n",
          "reproducer <subcommand> [<subcommand-options>]") {
  LoadSubCommand(
      "generate",
      CommandObjectSP(new CommandObjectReproducerGenerate(interpreter)));
  LoadSubCommand("status", CommandObjectSP(
                               new CommandObjectReproducerStatus(interpreter)));
  LoadSubCommand("dump",
                 CommandObjectSP(new CommandObjectReproducerDump(interpreter)));
  LoadSubCommand("verify", CommandObjectSP(
                               new CommandObjectReproducerVerify(interpreter)));
  LoadSubCommand("xcrash", CommandObjectSP(
                               new CommandObjectReproducerXCrash(interpreter)));
}

CommandObjectReproducer::~CommandObjectReproducer() = default;
