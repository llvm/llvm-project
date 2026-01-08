//===-- SBDebugger.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBDEBUGGER_H
#define LLDB_API_SBDEBUGGER_H

#include <cstdio>

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBPlatform.h"
#include "lldb/API/SBStructuredData.h"

namespace lldb_private {
class CommandPluginInterfaceImplementation;
class SystemInitializerFull;
namespace python {
class SWIGBridge;
}
} // namespace lldb_private

namespace lldb {

#ifndef SWIG
class LLDB_API SBInputReader {
public:
  SBInputReader() = default;
  ~SBInputReader() = default;

  SBError Initialize(lldb::SBDebugger &sb_debugger,
                     unsigned long (*callback)(void *, lldb::SBInputReader *,
                                               lldb::InputReaderAction,
                                               char const *, unsigned long),
                     void *a, lldb::InputReaderGranularity b, char const *c,
                     char const *d, bool e);
  void SetIsDone(bool);
  bool IsActive() const;
};
#endif

class LLDB_API SBDebugger {
public:
  /// Broadcast bit definitions for the SBDebugger.
  FLAGS_ANONYMOUS_ENUM() {
    eBroadcastBitProgress = lldb::DebuggerBroadcastBit::eBroadcastBitProgress,
    eBroadcastBitWarning = lldb::DebuggerBroadcastBit::eBroadcastBitWarning,
    eBroadcastBitError = lldb::DebuggerBroadcastBit::eBroadcastBitError,
    eBroadcastBitProgressCategory =
        lldb::DebuggerBroadcastBit::eBroadcastBitProgressCategory,
    eBroadcastBitExternalProgress =
        lldb::DebuggerBroadcastBit::eBroadcastBitExternalProgress,
    eBroadcastBitExternalProgressCategory =
        lldb::DebuggerBroadcastBit::eBroadcastBitExternalProgressCategory,
  };

  /// Default constructor creates an invalid SBDebugger instance.
  SBDebugger();

  SBDebugger(const lldb::SBDebugger &rhs);

  ~SBDebugger();

  /// Get the broadcaster class name.
  static const char *GetBroadcasterClass();

  /// Check if a specific language is supported by LLDB.
  static bool SupportsLanguage(lldb::LanguageType language);

  /// Get the broadcaster that allows subscribing to events from this
  /// debugger.
  lldb::SBBroadcaster GetBroadcaster();

  /// Get progress data from a SBEvent whose type is eBroadcastBitProgress.
  ///
  /// \param [in] event
  ///   The event to extract the progress information from.
  ///
  /// \param [out] progress_id
  ///   The unique integer identifier for the progress to report.
  ///
  /// \param [out] completed
  ///   The amount of work completed. If \a completed is zero, then this event
  ///   is a progress started event. If \a completed is equal to \a total, then
  ///   this event is a progress end event. Otherwise completed indicates the
  ///   current progress update.
  ///
  /// \param [out] total
  ///   The total amount of work units that need to be completed. If this value
  ///   is UINT64_MAX, then an indeterminate progress indicator should be
  ///   displayed.
  ///
  /// \param [out] is_debugger_specific
  ///   Set to true if this progress is specific to this debugger only. Many
  ///   progress events are not specific to a debugger instance, like any
  ///   progress events for loading information in modules since LLDB has a
  ///   global module cache that all debuggers use.
  ///
  /// \return The message for the progress. If the returned value is NULL, then
  ///   \a event was not a eBroadcastBitProgress event.
#ifdef SWIG
  static const char *GetProgressFromEvent(const lldb::SBEvent &event,
                                          uint64_t &OUTPUT,
                                          uint64_t &OUTPUT, uint64_t &OUTPUT,
                                          bool &OUTPUT);
#else
  static const char *GetProgressFromEvent(const lldb::SBEvent &event,
                                          uint64_t &progress_id,
                                          uint64_t &completed, uint64_t &total,
                                          bool &is_debugger_specific);
#endif

  /// Get progress data from an event.
  static lldb::SBStructuredData
  GetProgressDataFromEvent(const lldb::SBEvent &event);

  /// Get diagnostic information from an event.
  static lldb::SBStructuredData
  GetDiagnosticFromEvent(const lldb::SBEvent &event);

  /// Assignment operator.
  lldb::SBDebugger &operator=(const lldb::SBDebugger &rhs);

  /// Initialize LLDB and its subsystems.
  ///
  /// This function should be called before any other LLDB functions. It
  /// initializes all required subsystems for proper LLDB functionality.
  static void Initialize();

  /// Initialize the LLDB debugger subsystem with error handling.
  ///
  /// Similar to Initialize(), but returns an error if initialization fails.
  static lldb::SBError InitializeWithErrorHandling();

  /// Configure LLDB to print a stack trace when it crashes.
  static void PrintStackTraceOnError();

  /// Configure LLDB to print diagnostic information when it crashes.
  static void PrintDiagnosticsOnError();

  /// Terminate LLDB and its subsystems.
  ///
  /// This should be called when LLDB is no longer needed.
  static void Terminate();

  /// Create a new debugger instance (deprecated).
  LLDB_DEPRECATED_FIXME("Use one of the other Create variants", "Create(bool)")
  static lldb::SBDebugger Create();

  /// Create a new debugger instance.
  ///
  /// If source_init_files is true, the debugger will source .lldbinit files
  /// from the home directory and current directory.
  static lldb::SBDebugger Create(bool source_init_files);

  /// Create a new debugger instance with a custom log handler and user data
  /// passed to the log callback.
  ///
  /// If source_init_files is true, the debugger will source .lldbinit files
  /// from the home directory and current directory.
  static lldb::SBDebugger Create(bool source_init_files,
                                 lldb::LogOutputCallback log_callback,
                                 void *baton);

  /// Destroy a debugger instance.
  static void Destroy(lldb::SBDebugger &debugger);

  /// Notify the debugger that system memory pressure has been detected.
  ///
  /// This can be used to free up memory resources by clearing caches.
  static void MemoryPressureDetected();

  /// Check if this is a valid SBDebugger object.
  explicit operator bool() const;

  /// Check if this is a valid SBDebugger object.
  bool IsValid() const;

  /// Clear this debugger instance.
  ///
  /// This will close all IO handlers and reset the debugger to its initial
  /// state.
  void Clear();

  /// Get debugger settings as structured data.
  ///
  /// Client can specify empty string or null to get all settings.
  ///
  /// Example usages:
  /// lldb::SBStructuredData settings = debugger.GetSetting();
  /// lldb::SBStructuredData settings = debugger.GetSetting(nullptr);
  /// lldb::SBStructuredData settings = debugger.GetSetting("");
  /// lldb::SBStructuredData settings = debugger.GetSetting("target.arg0");
  /// lldb::SBStructuredData settings = debugger.GetSetting("target");
  lldb::SBStructuredData GetSetting(const char *setting = nullptr);

  /// Set whether the debugger should run in asynchronous mode.
  ///
  /// When in asynchronous mode, events are processed on a background thread.
  void SetAsync(bool b);

  /// Get whether the debugger is running in asynchronous mode.
  bool GetAsync();

  /// Set whether to skip loading .lldbinit files.
  void SkipLLDBInitFiles(bool b);

  /// Set whether to skip loading application-specific .lldbinit files.
  void SkipAppInitFiles(bool b);

#ifndef SWIG
  /// Set the input file handle for the debugger.
  void SetInputFileHandle(FILE *f, bool transfer_ownership);

  /// Set the output file handle for the debugger.
  void SetOutputFileHandle(FILE *f, bool transfer_ownership);

  /// Set the error file handle for the debugger.
  void SetErrorFileHandle(FILE *f, bool transfer_ownership);
#endif

#ifndef SWIG
  /// Get the input file handle for the debugger.
  FILE *GetInputFileHandle();

  /// Get the output file handle for the debugger.
  FILE *GetOutputFileHandle();

  /// Get the error file handle for the debugger.
  FILE *GetErrorFileHandle();
#endif

  /// Set the input from a string.
  SBError SetInputString(const char *data);

  /// Set the input file for the debugger.
  SBError SetInputFile(SBFile file);

  /// Set the output file for the debugger.
  SBError SetOutputFile(SBFile file);

  /// Set the error file for the debugger.
  SBError SetErrorFile(SBFile file);

  /// Set the input file for the debugger using a FileSP.
  SBError SetInputFile(FileSP file);

  /// Set the output file for the debugger using a FileSP.
  SBError SetOutputFile(FileSP file);

  /// Set the error file for the debugger using a FileSP.
  SBError SetErrorFile(FileSP file);

  /// Get the input file for the debugger.
  SBFile GetInputFile();

  /// Get the output file for the debugger.
  SBFile GetOutputFile();

  /// Get the error file for the debugger.
  SBFile GetErrorFile();

  /// Save the current terminal state.
  ///
  /// This should be called before modifying terminal settings.
  void SaveInputTerminalState();

  /// Restore the previously saved terminal state.
  void RestoreInputTerminalState();

  /// Get the command interpreter for this debugger.
  lldb::SBCommandInterpreter GetCommandInterpreter();

  /// Execute a command in the command interpreter.
  void HandleCommand(const char *command);

  /// Request an interrupt of the current operation.
  void RequestInterrupt();

  /// Cancel a previously requested interrupt.
  void CancelInterruptRequest();

  /// Check if an interrupt has been requested.
  bool InterruptRequested();

  /// Get the listener associated with this debugger.
  lldb::SBListener GetListener();

#ifndef SWIG
  /// Handle a process event (deprecated).
  LLDB_DEPRECATED_FIXME(
      "Use HandleProcessEvent(const SBProcess &, const SBEvent &, SBFile, "
      "SBFile) or HandleProcessEvent(const SBProcess &, const SBEvent &, "
      "FileSP, FileSP)",
      "HandleProcessEvent(const SBProcess &, const SBEvent &, SBFile, SBFile)")
  void HandleProcessEvent(const lldb::SBProcess &process,
                          const lldb::SBEvent &event, FILE *out, FILE *err);
#endif

  /// Handle a process event.
  void HandleProcessEvent(const lldb::SBProcess &process,
                          const lldb::SBEvent &event, SBFile out, SBFile err);

#ifdef SWIG
  /// Handle a process event using FileSP objects.
  void HandleProcessEvent(const lldb::SBProcess &process,
                          const lldb::SBEvent &event, FileSP BORROWED, FileSP BORROWED);
#else
  /// Handle a process event using FileSP objects.
  void HandleProcessEvent(const lldb::SBProcess &process,
                          const lldb::SBEvent &event, FileSP out, FileSP err);
#endif

  /// Create a target with the specified parameters.
  lldb::SBTarget CreateTarget(const char *filename, const char *target_triple,
                              const char *platform_name,
                              bool add_dependent_modules, lldb::SBError &error);

  /// Create a target with the specified file and target triple.
  lldb::SBTarget CreateTargetWithFileAndTargetTriple(const char *filename,
                                                     const char *target_triple);

  /// Create a target with the specified file and architecture.
  lldb::SBTarget CreateTargetWithFileAndArch(const char *filename,
                                             const char *archname);

  /// Create a target with the specified file.
  lldb::SBTarget CreateTarget(const char *filename);

  /// Get the dummy target.
  ///
  /// The dummy target is used when no target is available.
  lldb::SBTarget GetDummyTarget();

#ifndef SWIG
  /// Dispatch telemetry data from client to server.
  ///
  /// This is used to send telemetry data from the client to the server if
  /// client-telemetry is enabled. If not enabled, the data is ignored.
  void DispatchClientTelemetry(const lldb::SBStructuredData &data);
#endif

  /// Delete a target from the debugger.
  bool DeleteTarget(lldb::SBTarget &target);

  /// Get a target by index.
  lldb::SBTarget GetTargetAtIndex(uint32_t idx);

  /// Get the index of a target.
  uint32_t GetIndexOfTarget(lldb::SBTarget target);

  /// Find a target with the specified process ID.
  lldb::SBTarget FindTargetWithProcessID(lldb::pid_t pid);

  /// Find a target with the specified file and architecture.
  lldb::SBTarget FindTargetWithFileAndArch(const char *filename,
                                           const char *arch);

  /// Find a target with the specified unique ID.
  lldb::SBTarget FindTargetByGloballyUniqueID(lldb::user_id_t id);

  /// Get the number of targets in the debugger.
  uint32_t GetNumTargets();

  /// Get the currently selected target.
  lldb::SBTarget GetSelectedTarget();

  /// Set the selected target.
  void SetSelectedTarget(SBTarget &target);

  /// Get the selected platform.
  lldb::SBPlatform GetSelectedPlatform();

  /// Set the selected platform.
  void SetSelectedPlatform(lldb::SBPlatform &platform);

  /// Get the number of currently active platforms.
  uint32_t GetNumPlatforms();

  /// Get one of the currently active platforms.
  lldb::SBPlatform GetPlatformAtIndex(uint32_t idx);

  /// Get the number of available platforms.
  ///
  /// The return value should match the number of entries output by the
  /// "platform list" command.
  uint32_t GetNumAvailablePlatforms();

  /// Get information about the available platform at the given index as
  /// structured data.
  lldb::SBStructuredData GetAvailablePlatformInfoAtIndex(uint32_t idx);

  /// Get the source manager for this debugger.
  lldb::SBSourceManager GetSourceManager();

  /// Set the current platform by name.
  lldb::SBError SetCurrentPlatform(const char *platform_name);

  /// Set the SDK root for the current platform.
  bool SetCurrentPlatformSDKRoot(const char *sysroot);

  /// Set whether to use an external editor.
  bool SetUseExternalEditor(bool input);

  /// Get whether an external editor is being used.
  bool GetUseExternalEditor();

  /// Set whether to use color in output.
  bool SetUseColor(bool use_color);

  /// Get whether color is being used in output.
  bool GetUseColor() const;

  /// Set whether to show inline diagnostics.
  bool SetShowInlineDiagnostics(bool b);

  /// Set whether to use the source cache.
  bool SetUseSourceCache(bool use_source_cache);

  /// Get whether the source cache is being used.
  bool GetUseSourceCache() const;

  /// Get the default architecture.
  static bool GetDefaultArchitecture(char *arch_name, size_t arch_name_len);

  /// Set the default architecture.
  static bool SetDefaultArchitecture(const char *arch_name);

  /// Get the scripting language by name.
  lldb::ScriptLanguage GetScriptingLanguage(const char *script_language_name);

  /// Get information about a script interpreter as structured data.
  SBStructuredData GetScriptInterpreterInfo(ScriptLanguage language);

  /// Get the LLDB version string.
  static const char *GetVersionString();

  /// Convert a state type to a string.
  static const char *StateAsCString(lldb::StateType state);

  /// Get the build configuration as structured data.
  static SBStructuredData GetBuildConfiguration();

  /// Check if a state is a running state.
  static bool StateIsRunningState(lldb::StateType state);

  /// Check if a state is a stopped state.
  static bool StateIsStoppedState(lldb::StateType state);

  /// Enable logging for a specific channel and category.
  bool EnableLog(const char *channel, const char **categories);

  /// Set a callback for log output.
  void SetLoggingCallback(lldb::LogOutputCallback log_callback, void *baton);

  /// Set a callback for when the debugger is destroyed (deprecated).
  LLDB_DEPRECATED_FIXME("Use AddDestroyCallback and RemoveDestroyCallback",
                        "AddDestroyCallback")
  void SetDestroyCallback(lldb::SBDebuggerDestroyCallback destroy_callback,
                          void *baton);

  /// Add a callback for when the debugger is destroyed. Returns a token that
  /// can be used to remove the callback.
  lldb::callback_token_t
  AddDestroyCallback(lldb::SBDebuggerDestroyCallback destroy_callback,
                     void *baton);

  /// Remove a destroy callback.
  bool RemoveDestroyCallback(lldb::callback_token_t token);

#ifndef SWIG
  /// Dispatch input to the debugger (deprecated).
  LLDB_DEPRECATED_FIXME("Use DispatchInput(const void *, size_t)",
                        "DispatchInput(const void *, size_t)")
  void DispatchInput(void *baton, const void *data, size_t data_len);
#endif

  /// Dispatch input to the debugger.
  void DispatchInput(const void *data, size_t data_len);

  /// Interrupt the current input dispatch.
  void DispatchInputInterrupt();

  /// Signal end-of-file to the current input dispatch.
  void DispatchInputEndOfFile();

#ifndef SWIG
  /// Push an input reader onto the IO handler stack.
  void PushInputReader(lldb::SBInputReader &reader);
#endif

  /// Get the instance name of this debugger.
  const char *GetInstanceName();

  /// Find a debugger by ID. Returns an invalid debugger if not found.
  static SBDebugger FindDebuggerWithID(int id);

  /// Set an internal variable.
  static lldb::SBError SetInternalVariable(const char *var_name,
                                           const char *value,
                                           const char *debugger_instance_name);

  /// Get the value of an internal variable.
  static lldb::SBStringList
  GetInternalVariableValue(const char *var_name,
                           const char *debugger_instance_name);

  /// Get a description of this debugger.
  bool GetDescription(lldb::SBStream &description);

  /// Get the terminal width.
  uint32_t GetTerminalWidth() const;

  /// Set the terminal width.
  void SetTerminalWidth(uint32_t term_width);

  /// Get the terminal height.
  uint32_t GetTerminalHeight() const;

  /// Set the terminal height.
  void SetTerminalHeight(uint32_t term_height);

  /// Get the unique ID of this debugger.
  lldb::user_id_t GetID();

  /// Get the command prompt string.
  const char *GetPrompt() const;

  /// Set the command prompt string.
  void SetPrompt(const char *prompt);

  /// Get the path to the reproducer.
  const char *GetReproducerPath() const;

  /// Get the current scripting language.
  lldb::ScriptLanguage GetScriptLanguage() const;

  /// Set the current scripting language.
  void SetScriptLanguage(lldb::ScriptLanguage script_lang);

  /// Get the current REPL language.
  lldb::LanguageType GetREPLLanguage() const;

  /// Set the current REPL language.
  void SetREPLLanguage(lldb::LanguageType repl_lang);

  /// Get whether to close input on EOF (deprecated).
  LLDB_DEPRECATED("SBDebugger::GetCloseInputOnEOF() is deprecated.")
  bool GetCloseInputOnEOF() const;

  /// Set whether to close input on EOF (deprecated).
  LLDB_DEPRECATED("SBDebugger::SetCloseInputOnEOF() is deprecated.")
  void SetCloseInputOnEOF(bool b);

  /// Get a type category by name.
  SBTypeCategory GetCategory(const char *category_name);

  /// Get a type category by language.
  SBTypeCategory GetCategory(lldb::LanguageType lang_type);

  /// Create a new type category.
  SBTypeCategory CreateCategory(const char *category_name);

  /// Delete a type category.
  bool DeleteCategory(const char *category_name);

  /// Get the number of type categories.
  uint32_t GetNumCategories();

  /// Get a type category by index.
  SBTypeCategory GetCategoryAtIndex(uint32_t index);

  /// Get the default type category.
  SBTypeCategory GetDefaultCategory();

  /// Get the format for a type.
  SBTypeFormat GetFormatForType(SBTypeNameSpecifier type_name_spec);

  /// Get the summary for a type.
  SBTypeSummary GetSummaryForType(SBTypeNameSpecifier type_name_spec);

  /// Get the filter for a type.
  SBTypeFilter GetFilterForType(SBTypeNameSpecifier type_name_spec);

  /// Get the synthetic for a type.
  SBTypeSynthetic GetSyntheticForType(SBTypeNameSpecifier type_name_spec);

  /// Clear collected statistics for targets belonging to this debugger.
  ///
  /// This includes clearing symbol table and debug info parsing/index time for
  /// all modules, breakpoint resolve time, and target statistics.
  void ResetStatistics();

#ifndef SWIG
  /// Run the command interpreter.
  ///
  /// \param[in] auto_handle_events
  ///   If true, automatically handle resulting events. This takes precedence
  ///   and overrides the corresponding option in
  ///   SBCommandInterpreterRunOptions.
  ///
  /// \param[in] spawn_thread
  ///   If true, start a new thread for IO handling. This takes precedence and
  ///   overrides the corresponding option in SBCommandInterpreterRunOptions.
  void RunCommandInterpreter(bool auto_handle_events, bool spawn_thread);
#endif

  /// Run the command interpreter with options.
  ///
  /// \param[in] auto_handle_events
  ///   If true, automatically handle resulting events. This takes precedence
  ///   and overrides the corresponding option in
  ///   SBCommandInterpreterRunOptions.
  ///
  /// \param[in] spawn_thread
  ///   If true, start a new thread for IO handling. This takes precedence and
  ///   overrides the corresponding option in SBCommandInterpreterRunOptions.
  ///
  /// \param[in] options
  ///   Parameter collection of type SBCommandInterpreterRunOptions.
  ///
  /// \param[out] num_errors
  ///   The number of errors.
  ///
  /// \param[out] quit_requested
  ///   Whether a quit was requested.
  ///
  /// \param[out] stopped_for_crash
  ///   Whether the interpreter stopped for a crash.
#ifdef SWIG
  %apply int& INOUT { int& num_errors };
  %apply bool& INOUT { bool& quit_requested };
  %apply bool& INOUT { bool& stopped_for_crash };
#endif
  void RunCommandInterpreter(bool auto_handle_events, bool spawn_thread,
                             SBCommandInterpreterRunOptions &options,
                             int &num_errors, bool &quit_requested,
                             bool &stopped_for_crash);

#ifndef SWIG
  /// Run the command interpreter with options and return a result object.
  SBCommandInterpreterRunResult
  RunCommandInterpreter(const SBCommandInterpreterRunOptions &options);
#endif

  /// Run a REPL (Read-Eval-Print Loop) for the specified language.
  SBError RunREPL(lldb::LanguageType language, const char *repl_options);

  /// Load a trace from a trace description file.
  ///
  /// This will create Targets, Processes and Threads based on the contents of
  /// the file.
  ///
  /// \param[out] error
  ///   An error if the trace could not be created.
  ///
  /// \param[in] trace_description_file
  ///   The file containing the necessary information to load the trace.
  ///
  /// \return
  ///   An SBTrace object representing the loaded trace.
  SBTrace LoadTraceFromFile(SBError &error,
                            const SBFileSpec &trace_description_file);

protected:
  friend class lldb_private::CommandPluginInterfaceImplementation;
  friend class lldb_private::python::SWIGBridge;
  friend class lldb_private::SystemInitializerFull;

  SBDebugger(const lldb::DebuggerSP &debugger_sp);

private:
  friend class SBCommandInterpreter;
  friend class SBInputReader;
  friend class SBListener;
  friend class SBProcess;
  friend class SBSourceManager;
  friend class SBStructuredData;
  friend class SBPlatform;
  friend class SBTarget;
  friend class SBTrace;
  friend class SBProgress;

  lldb::SBTarget FindTargetWithLLDBProcess(const lldb::ProcessSP &processSP);

  void reset(const lldb::DebuggerSP &debugger_sp);

  lldb_private::Debugger *get() const;

  lldb_private::Debugger &ref() const;

  const lldb::DebuggerSP &get_sp() const;

  lldb::DebuggerSP m_opaque_sp;

}; // class SBDebugger

} // namespace lldb

#endif // LLDB_API_SBDEBUGGER_H
