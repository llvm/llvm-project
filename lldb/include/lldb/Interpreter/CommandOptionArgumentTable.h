//===-- CommandOptionArgumentTable.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_COMMANDOPTIONARGUMENTTABLE_H
#define LLDB_INTERPRETER_COMMANDOPTIONARGUMENTTABLE_H

#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

static constexpr OptionEnumValueElement g_corefile_save_style[] = {
    {lldb::eSaveCoreFull, "full", "Create a core file with all memory saved"},
    {lldb::eSaveCoreDirtyOnly, "modified-memory",
     "Create a corefile with only modified memory saved"},
    {lldb::eSaveCoreStackOnly, "stack",
     "Create a corefile with only stack  memory saved"},
};

static constexpr OptionEnumValueElement g_description_verbosity_type[] = {
    {
        eLanguageRuntimeDescriptionDisplayVerbosityCompact,
        "compact",
        "Only show the description string",
    },
    {
        eLanguageRuntimeDescriptionDisplayVerbosityFull,
        "full",
        "Show the full output, including persistent variable's name and type",
    },
};

static constexpr OptionEnumValueElement g_sort_option_enumeration[] = {
    {
        eSortOrderNone,
        "none",
        "No sorting, use the original symbol table order.",
    },
    {
        eSortOrderByAddress,
        "address",
        "Sort output by symbol address.",
    },
    {
        eSortOrderByName,
        "name",
        "Sort output by symbol name.",
    },
};

// Note that the negation in the argument name causes a slightly confusing
// mapping of the enum values.
static constexpr OptionEnumValueElement g_dependents_enumeration[] = {
    {
        eLoadDependentsDefault,
        "default",
        "Only load dependents when the target is an executable.",
    },
    {
        eLoadDependentsNo,
        "true",
        "Don't load dependents, even if the target is an executable.",
    },
    {
        eLoadDependentsYes,
        "false",
        "Load dependents, even if the target is not an executable.",
    },
};

// FIXME: "script-type" needs to have its contents determined dynamically, so
// somebody can add a new scripting language to lldb and have it pickable here
// without having to change this enumeration by hand and rebuild lldb proper.
static constexpr OptionEnumValueElement g_script_option_enumeration[] = {
    {
        lldb::eScriptLanguageNone,
        "command",
        "Commands are in the lldb command interpreter language",
    },
    {
        lldb::eScriptLanguagePython,
        "python",
        "Commands are in the Python language.",
    },
    {
        lldb::eScriptLanguageLua,
        "lua",
        "Commands are in the Lua language.",
    },
    {
        lldb::eScriptLanguageNone,
        "default",
        "Commands are in the default scripting language.",
    },
};

static constexpr OptionEnumValueElement g_log_handler_type[] = {
    {
        eLogHandlerDefault,
        "default",
        "Use the default (stream) log handler",
    },
    {
        eLogHandlerStream,
        "stream",
        "Write log messages to the debugger output stream or to a file if one "
        "is specified. A buffer size (in bytes) can be specified with -b. If "
        "no buffer size is specified the output is unbuffered.",
    },
    {
        eLogHandlerCircular,
        "circular",
        "Write log messages to a fixed size circular buffer. A buffer size "
        "(number of messages) must be specified with -b.",
    },
    {
        eLogHandlerSystem,
        "os",
        "Write log messages to the operating system log.",
    },
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

static constexpr OptionEnumValueElement g_script_synchro_type[] = {
    {
        eScriptedCommandSynchronicitySynchronous,
        "synchronous",
        "Run synchronous",
    },
    {
        eScriptedCommandSynchronicityAsynchronous,
        "asynchronous",
        "Run asynchronous",
    },
    {
        eScriptedCommandSynchronicityCurrentValue,
        "current",
        "Do not alter current setting",
    },
};

static constexpr OptionEnumValueElement g_running_mode[] = {
    {lldb::eOnlyThisThread, "this-thread", "Run only this thread"},
    {lldb::eAllThreads, "all-threads", "Run all threads"},
    {lldb::eOnlyDuringStepping, "while-stepping",
     "Run only this thread while stepping"},
};

llvm::StringRef RegisterNameHelpTextCallback();
llvm::StringRef BreakpointIDHelpTextCallback();
llvm::StringRef BreakpointIDRangeHelpTextCallback();
llvm::StringRef BreakpointNameHelpTextCallback();
llvm::StringRef GDBFormatHelpTextCallback();
llvm::StringRef FormatHelpTextCallback();
llvm::StringRef LanguageTypeHelpTextCallback();
llvm::StringRef SummaryStringHelpTextCallback();
llvm::StringRef ExprPathHelpTextCallback();
llvm::StringRef arch_helper();

static constexpr CommandObject::ArgumentTableEntry g_argument_table[] = {
    // clang-format off
    { lldb::eArgTypeAddress, "address", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "A valid address in the target program's execution space." },
    { lldb::eArgTypeAddressOrExpression, "address-expression", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "An expression that resolves to an address." },
    { lldb::eArgTypeAliasName, "alias-name", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "The name of an abbreviation (alias) for a debugger command." },
    { lldb::eArgTypeAliasOptions, "options-for-aliased-command", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Command options to be used as part of an alias (abbreviation) definition.  (See 'help commands alias' for more information.)" },
    { lldb::eArgTypeArchitecture, "arch", CommandCompletions::eArchitectureCompletion, {}, { arch_helper, true }, "The architecture name, e.g. i386 or x86_64." },
    { lldb::eArgTypeBoolean, "boolean", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "A Boolean value: 'true' or 'false'" },
    { lldb::eArgTypeBreakpointID, "breakpt-id", CommandCompletions::eNoCompletion, {}, { BreakpointIDHelpTextCallback, false }, nullptr },
    { lldb::eArgTypeBreakpointIDRange, "breakpt-id-list", CommandCompletions::eNoCompletion, {}, { BreakpointIDRangeHelpTextCallback, false }, nullptr },
    { lldb::eArgTypeBreakpointName, "breakpoint-name", CommandCompletions::eBreakpointNameCompletion, {}, { BreakpointNameHelpTextCallback, false }, nullptr },
    { lldb::eArgTypeByteSize, "byte-size", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Number of bytes to use." },
    { lldb::eArgTypeClassName, "class-name", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Then name of a class from the debug information in the program." },
    { lldb::eArgTypeCommandName, "cmd-name", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "A debugger command (may be multiple words), without any options or arguments." },
    { lldb::eArgTypeCount, "count", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "An unsigned integer." },
    { lldb::eArgTypeDescriptionVerbosity, "description-verbosity", CommandCompletions::eNoCompletion, g_description_verbosity_type, { nullptr, false }, "How verbose the output of 'po' should be." },
    { lldb::eArgTypeDirectoryName, "directory", CommandCompletions::eDiskDirectoryCompletion, {}, { nullptr, false }, "A directory name." },
    { lldb::eArgTypeDisassemblyFlavor, "disassembly-flavor", CommandCompletions::eDisassemblyFlavorCompletion, {}, { nullptr, false }, "A disassembly flavor recognized by your disassembly plugin.  Currently the only valid options are \"att\" and \"intel\" for Intel targets" },
    { lldb::eArgTypeEndAddress, "end-address", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Help text goes here." },
    { lldb::eArgTypeExpression, "expr", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Help text goes here." },
    { lldb::eArgTypeExpressionPath, "expr-path", CommandCompletions::eNoCompletion, {}, { ExprPathHelpTextCallback, true }, nullptr },
    { lldb::eArgTypeExprFormat, "expression-format", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "[ [bool|b] | [bin] | [char|c] | [oct|o] | [dec|i|d|u] | [hex|x] | [float|f] | [cstr|s] ]" },
    { lldb::eArgTypeFileLineColumn, "linespec", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "A source specifier in the form file:line[:column]" },
    { lldb::eArgTypeFilename, "filename", CommandCompletions::eDiskFileCompletion, {}, { nullptr, false }, "The name of a file (can include path)." },
    { lldb::eArgTypeFormat, "format", CommandCompletions::eNoCompletion, {}, { FormatHelpTextCallback, true }, nullptr },
    { lldb::eArgTypeFrameIndex, "frame-index", CommandCompletions::eFrameIndexCompletion, {}, { nullptr, false }, "Index into a thread's list of frames." },
    { lldb::eArgTypeFullName, "fullname", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Help text goes here." },
    { lldb::eArgTypeFunctionName, "function-name", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "The name of a function." },
    { lldb::eArgTypeFunctionOrSymbol, "function-or-symbol", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "The name of a function or symbol." },
    { lldb::eArgTypeGDBFormat, "gdb-format", CommandCompletions::eNoCompletion, {}, { GDBFormatHelpTextCallback, true }, nullptr },
    { lldb::eArgTypeHelpText, "help-text", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Text to be used as help for some other entity in LLDB" },
    { lldb::eArgTypeIndex, "index", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "An index into a list." },
    { lldb::eArgTypeLanguage, "source-language", CommandCompletions::eTypeLanguageCompletion, {}, { LanguageTypeHelpTextCallback, true }, nullptr },
    { lldb::eArgTypeLineNum, "linenum", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Line number in a source file." },
    { lldb::eArgTypeLogCategory, "log-category", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "The name of a category within a log channel, e.g. all (try \"log list\" to see a list of all channels and their categories." },
    { lldb::eArgTypeLogChannel, "log-channel", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "The name of a log channel, e.g. process.gdb-remote (try \"log list\" to see a list of all channels and their categories)." },
    { lldb::eArgTypeMethod, "method", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "A C++ method name." },
    { lldb::eArgTypeName, "name", CommandCompletions::eTypeCategoryNameCompletion, {}, { nullptr, false }, "Help text goes here." },
    { lldb::eArgTypeNewPathPrefix, "new-path-prefix", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Help text goes here." },
    { lldb::eArgTypeNumLines, "num-lines", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "The number of lines to use." },
    { lldb::eArgTypeNumberPerLine, "number-per-line", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "The number of items per line to display." },
    { lldb::eArgTypeOffset, "offset", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Help text goes here." },
    { lldb::eArgTypeOldPathPrefix, "old-path-prefix", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Help text goes here." },
    { lldb::eArgTypeOneLiner, "one-line-command", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "A command that is entered as a single line of text." },
    { lldb::eArgTypePath, "path", CommandCompletions::eDiskFileCompletion, {}, { nullptr, false }, "Path." },
    { lldb::eArgTypePermissionsNumber, "perms-numeric", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Permissions given as an octal number (e.g. 755)." },
    { lldb::eArgTypePermissionsString, "perms=string", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Permissions given as a string value (e.g. rw-r-xr--)." },
    { lldb::eArgTypePid, "pid", CommandCompletions::eProcessIDCompletion, {}, { nullptr, false }, "The process ID number." },
    { lldb::eArgTypePlugin, "plugin", CommandCompletions::eProcessPluginCompletion, {}, { nullptr, false }, "Help text goes here." },
    { lldb::eArgTypeProcessName, "process-name", CommandCompletions::eProcessNameCompletion, {}, { nullptr, false }, "The name of the process." },
    { lldb::eArgTypePythonClass, "python-class", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "The name of a Python class." },
    { lldb::eArgTypePythonFunction, "python-function", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "The name of a Python function." },
    { lldb::eArgTypePythonScript, "python-script", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Source code written in Python." },
    { lldb::eArgTypeQueueName, "queue-name", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "The name of the thread queue." },
    { lldb::eArgTypeRegisterName, "register-name", CommandCompletions::eNoCompletion, {}, { RegisterNameHelpTextCallback, true }, nullptr },
    { lldb::eArgTypeRegularExpression, "regular-expression", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "A POSIX-compliant extended regular expression." },
    { lldb::eArgTypeRunArgs, "run-args", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Arguments to be passed to the target program when it starts executing." },
    { lldb::eArgTypeRunMode, "run-mode", CommandCompletions::eNoCompletion, g_running_mode, { nullptr, false }, "Help text goes here." },
    { lldb::eArgTypeScriptedCommandSynchronicity, "script-cmd-synchronicity", CommandCompletions::eNoCompletion, g_script_synchro_type, { nullptr, false }, "The synchronicity to use to run scripted commands with regard to LLDB event system." },
    { lldb::eArgTypeScriptLang, "script-language", CommandCompletions::eNoCompletion, g_script_option_enumeration, { nullptr, false }, "The scripting language to be used for script-based commands.  Supported languages are python and lua." },
    { lldb::eArgTypeSearchWord, "search-word", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Any word of interest for search purposes." },
    { lldb::eArgTypeSelector, "selector", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "An Objective-C selector name." },
    { lldb::eArgTypeSettingIndex, "setting-index", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "An index into a settings variable that is an array (try 'settings list' to see all the possible settings variables and their types)." },
    { lldb::eArgTypeSettingKey, "setting-key", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "A key into a settings variables that is a dictionary (try 'settings list' to see all the possible settings variables and their types)." },
    { lldb::eArgTypeSettingPrefix, "setting-prefix", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "The name of a settable internal debugger variable up to a dot ('.'), e.g. 'target.process.'" },
    { lldb::eArgTypeSettingVariableName, "setting-variable-name", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "The name of a settable internal debugger variable.  Type 'settings list' to see a complete list of such variables." },
    { lldb::eArgTypeShlibName, "shlib-name", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "The name of a shared library." },
    { lldb::eArgTypeSourceFile, "source-file", CommandCompletions::eSourceFileCompletion, {}, { nullptr, false }, "The name of a source file.." },
    { lldb::eArgTypeSortOrder, "sort-order", CommandCompletions::eNoCompletion, g_sort_option_enumeration, { nullptr, false }, "Specify a sort order when dumping lists." },
    { lldb::eArgTypeStartAddress, "start-address", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Help text goes here." },
    { lldb::eArgTypeSummaryString, "summary-string", CommandCompletions::eNoCompletion, {}, { SummaryStringHelpTextCallback, true }, nullptr },
    { lldb::eArgTypeSymbol, "symbol", CommandCompletions::eSymbolCompletion, {}, { nullptr, false }, "Any symbol name (function name, variable, argument, etc.)" },
    { lldb::eArgTypeThreadID, "thread-id", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Thread ID number." },
    { lldb::eArgTypeThreadIndex, "thread-index", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Index into the process' list of threads." },
    { lldb::eArgTypeThreadName, "thread-name", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "The thread's name." },
    { lldb::eArgTypeTypeName, "type-name", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "A type name." },
    { lldb::eArgTypeUnsignedInteger, "unsigned-integer", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "An unsigned integer." },
    { lldb::eArgTypeUnixSignal, "unix-signal", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "A valid Unix signal name or number (e.g. SIGKILL, KILL or 9)." },
    { lldb::eArgTypeVarName, "variable-name", CommandCompletions::eNoCompletion, {} ,{ nullptr, false }, "The name of a variable in your program." },
    { lldb::eArgTypeValue, "value", CommandCompletions::eNoCompletion, g_dependents_enumeration, { nullptr, false }, "A value could be anything, depending on where and how it is used." },
    { lldb::eArgTypeWidth, "width", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Help text goes here." },
    { lldb::eArgTypeNone, "none", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "No help available for this." },
    { lldb::eArgTypePlatform, "platform-name", CommandCompletions::ePlatformPluginCompletion, {}, { nullptr, false }, "The name of an installed platform plug-in . Type 'platform list' to see a complete list of installed platforms." },
    { lldb::eArgTypeWatchpointID, "watchpt-id", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Watchpoint IDs are positive integers." },
    { lldb::eArgTypeWatchpointIDRange, "watchpt-id-list", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "For example, '1-3' or '1 to 3'." },
    { lldb::eArgTypeWatchType, "watch-type", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Specify the type for a watchpoint." },
    { lldb::eArgRawInput, "raw-input", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Free-form text passed to a command without prior interpretation, allowing spaces without requiring quotes.  To pass arguments and free form text put two dashes ' -- ' between the last argument and any raw input." },
    { lldb::eArgTypeCommand, "command", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "An LLDB Command line command element." },
    { lldb::eArgTypeColumnNum, "column", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "Column number in a source file." },
    { lldb::eArgTypeModuleUUID, "module-uuid", CommandCompletions::eModuleUUIDCompletion, {}, { nullptr, false }, "A module UUID value." },
    { lldb::eArgTypeSaveCoreStyle, "corefile-style", CommandCompletions::eNoCompletion, g_corefile_save_style, { nullptr, false }, "The type of corefile that lldb will try to create, dependant on this target's capabilities." },
    { lldb::eArgTypeLogHandler, "log-handler", CommandCompletions::eNoCompletion, g_log_handler_type ,{ nullptr, false }, "The log handle that will be used to write out log messages." },
    { lldb::eArgTypeSEDStylePair, "substitution-pair", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "A sed-style pattern and target pair." },
    { lldb::eArgTypeRecognizerID, "frame-recognizer-id", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "The ID for a stack frame recognizer." },
    { lldb::eArgTypeConnectURL, "process-connect-url", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "A URL-style specification for a remote connection." },
    { lldb::eArgTypeTargetID, "target-id", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "The index ID for an lldb Target." },
    { lldb::eArgTypeStopHookID, "stop-hook-id", CommandCompletions::eNoCompletion, {}, { nullptr, false }, "The ID you receive when you create a stop-hook." },
    { lldb::eArgTypeReproducerProvider, "reproducer-provider", CommandCompletions::eNoCompletion, g_reproducer_provider_type, { nullptr, false }, "The reproducer provider." },
    { lldb::eArgTypeReproducerSignal, "reproducer-signal", CommandCompletions::eNoCompletion, g_reproducer_signaltype, { nullptr, false }, "The signal used to emulate a reproducer crash." },
    // clang-format on
};

static_assert((sizeof(g_argument_table) /
               sizeof(CommandObject::ArgumentTableEntry)) ==
                  lldb::eArgTypeLastArg,
              "number of elements in g_argument_table doesn't match "
              "CommandArgumentType enumeration");

} // namespace lldb_private

#endif // LLDB_INTERPRETER_COMMANDOPTIONARGUMENTTABLE_H
