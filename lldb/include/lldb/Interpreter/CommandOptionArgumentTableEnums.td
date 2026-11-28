include "../Core/EnumsBase.td"

def corefile_save_style : EnumDef<"lldb::eSaveCore", [
  EnumValue<"Full", "full", "Create a core file with all memory saved">,
  EnumValue<"DirtyOnly", "modified-memory",
    "Create a corefile with only modified memory saved">,
  EnumValue<"StackOnly", "stack",
    "Create a corefile with only stack  memory saved">,
]>;

def description_verbosity_type : EnumDef<"eLanguageRuntimeDescriptionDisplayVerbosity", [
  EnumValue<"Compact", "compact", "Only show the description string">,
  EnumValue<"Full", "full",
    "Show the full output, including persistent variable's name and type">,
]>;

def sort_option_enumeration : EnumDef<"eSortOrder", [
  EnumValue<"None", "none", "No sorting, use the original symbol table order.">,
  EnumValue<"ByAddress", "address", "Sort output by symbol address.">,
  EnumValue<"ByName", "name", "Sort output by symbol name.">,
  EnumValue<"BySize", "size", "Sort output by symbol byte size.">,
]>;


// Note that the negation in the argument name causes a slightly confusing
// mapping of the enum values.
def dependents_enumeration : EnumDef<"eLoadDependents", [
  EnumValue<"Default", "default",
    "Only load dependents when the target is an executable.">,
  EnumValue<"No", "true",
    "Don't load dependents, even if the target is an executable.">,
  EnumValue<"Yes", "false",
    "Load dependents, even if the target is not an executable.">,
]>;

// FIXME: "script-type" needs to have its contents determined dynamically, so
// somebody can add a new scripting language to lldb and have it pickable here
// without having to change this enumeration by hand and rebuild lldb proper.
def script_option_enumeration : EnumDef<"lldb::eScriptLanguage", [
  EnumValue<"None", "command",
    "Commands are in the lldb command interpreter language">,
  EnumValue<"Python", "python", "Commands are in the Python language.">,
  EnumValue<"Lua", "lua", "Commands are in the Lua language.">,
  EnumValue<"None", "default", "Commands are in the default scripting language.">,
]>;

def log_handler_type : EnumDef<"eLogHandler", [
  EnumValue<"Default", "default", "Use the default (stream) log handler">,
  EnumValue<"Stream", "stream",
    "Write log messages to the debugger output stream or to a file if one is specified. "
    "A buffer size (in bytes) can be specified with -b. "
    "If no buffer size is specified the output is unbuffered.">,
  EnumValue<"Circular", "circular",
    "Write log messages to a fixed size circular buffer. "
    "A buffer size (number of messages) must be specified with -b.">,
  EnumValue<"System", "os", "Write log messages to the operating system log.">,
]>;

def script_synchro_type : EnumDef<"eScriptedCommandSynchronicity", [
  EnumValue<"Synchronous", "synchronous", "Run synchronous">,
  EnumValue<"Asynchronous", "asynchronous", "Run asynchronous">,
  EnumValue<"CurrentValue", "current", "Do not alter current setting">,
]>;

def running_mode : EnumDef<"lldb::", [
  EnumValue<"eOnlyThisThread", "this-thread", "Run only this thread">,
  EnumValue<"eAllThreads", "all-threads", "Run all threads">,
  EnumValue<"eOnlyDuringStepping", "while-stepping",
    "Run only this thread while stepping">,
]>;

def exception_stage : EnumDef<"lldb::eExceptionStage", [
  EnumValue<"Throw", "throw", "Stop when the exception is thrown.">,
  EnumValue<"ReThrow", "re-throw", "Stop when the exception is re-thrown.">,
  EnumValue<"Catch", "catch", "Stop when the exception is caught.">,
]>;

def name_match_style : EnumDef<"lldb::eNameMatchStyle", [
  EnumValue<"Auto", "auto",
    "Match against the leaf nodes of the identifier, or against methods or selectors.">,
  EnumValue<"Full", "full", "Match the full identifier name.">,
  EnumValue<"Base", "base", "Match against the leaf node of the identifier.">,
  EnumValue<"Method", "method", "Match only against method names.">,
  EnumValue<"Selector", "selector", "Match only against selector names.">,
  EnumValue<"Regex", "regex", "Match the identifier using a regular expression.">,
]>;

def plugin_domain_values : EnumDef<"lldb::ePluginDomainKind", [
  EnumValue<"Global", "global", "Apply to all debugger instances.">,
  EnumValue<"Debugger", "debugger", "Apply to the current debugger instance.">,
  EnumValue<"Target", "target", "Apply to the current target.">,
]>;

def resolver_mask_values : EnumDef<"lldb::eResolver", [
  EnumValue<"FileAndLine", "file_and_line",
    "Matches file and line breakpoint resolvers." >,
  EnumValue<"Address", "address", "Matches address breakpoint resolvers.">,
  EnumValue<"Name", "name", "Matches symbol name breakpoint resolvers.">,
  EnumValue<"FileRegex", "source", "Matches source text pattern resolvers.">,
  EnumValue<"Python", "scripted", "Matches scripted breakpoint resolvers.">,
  EnumValue<"Exception", "exception",
    "Matches language exception breakpoint resolvers.">,
]>;

def completion_type : EnumDef<"lldb::", [
  EnumValue<"eNoCompletion", "none", "No completion.">,
  EnumValue<"eSourceFileCompletion", "source-file",
    "Completes to a source file.">,
  EnumValue<"eDiskFileCompletion", "disk-file", "Completes to a disk file.">,
  EnumValue<"eDiskDirectoryCompletion", "disk-directory",
    "Completes to a disk directory.">,
  EnumValue<"eSymbolCompletion", "symbol", "Completes to a symbol.">,
  EnumValue<"eModuleCompletion", "module", "Completes to a module.">,
  EnumValue<"eSettingsNameCompletion", "settings-name",
    "Completes to a settings name.">,
  EnumValue<"ePlatformPluginCompletion", "platform-plugin",
    "Completes to a platform plugin.">,
  EnumValue<"eArchitectureCompletion", "architecture",
    "Completes to a architecture.">,
  EnumValue<"eVariablePathCompletion", "variable-path",
    "Completes to a variable path.">,
  EnumValue<"eRegisterCompletion", "register", "Completes to a register.">,
  EnumValue<"eBreakpointCompletion", "breakpoint",
    "Completes to a breakpoint.">,
  EnumValue<"eProcessPluginCompletion", "process-plugin",
    "Completes to a process plugin.">,
  EnumValue<"eDisassemblyFlavorCompletion", "disassembly-flavor",
    "Completes to a disassembly flavor.">,
  EnumValue<"eTypeLanguageCompletion", "type-language",
    "Completes to a type language.">,
  EnumValue<"eFrameIndexCompletion", "frame-index",
    "Completes to a frame index.">,
  EnumValue<"eModuleUUIDCompletion", "module-uuid",
    "Completes to a module uuid.">,
  EnumValue<"eStopHookIDCompletion", "stophook-id",
    "Completes to a stophook id.">,
  EnumValue<"eThreadIndexCompletion", "thread-index",
    "Completes to a thread index.">,
  EnumValue<"eWatchpointIDCompletion", "watchpoint-id",
    "Completes to a watchpoint id.">,
  EnumValue<"eBreakpointNameCompletion", "breakpoint-name",
    "Completes to a breakpoint name.">,
  EnumValue<"eProcessIDCompletion", "process-id",
    "Completes to a process id.">,
  EnumValue<"eProcessNameCompletion", "process-name",
    "Completes to a process name.">,
  EnumValue<"eRemoteDiskFileCompletion", "remote-disk-file",
    "Completes to a remote disk file.">,
  EnumValue<"eRemoteDiskDirectoryCompletion", "remote-disk-directory",
    "Completes to a remote disk directory.">,
  EnumValue<"eTypeCategoryNameCompletion", "type-category-name",
    "Completes to a type category name.">,
  EnumValue<"eCustomCompletion", "custom", "Custom completion.">,
  EnumValue<"eThreadIDCompletion", "thread-id", "Completes to a thread ID.">,
]>;
