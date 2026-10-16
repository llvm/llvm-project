include "../../include/lldb/Core/EnumsBase.td"

def show_disassembly_enum_values : EnumDef<"lldb::eStopDisassemblyType", [
  EnumValue<"Never", "never",
    "Never show disassembly when displaying a stop context.">,
  EnumValue<"NoDebugInfo", "no-debuginfo",
    "Show disassembly when there is no debug information.">,
  EnumValue<"NoSource", "no-source",
    "Show disassembly when there is no source information, or the source file "
    "is missing when displaying a stop context.">,
  EnumValue<"Always", "always",
    "Always show disassembly when displaying a stop context.">,
]>;

def language_enumerators : EnumDef<"eScriptLanguage", [
  EnumValue<"None", "none", "Disable scripting languages.">,
  EnumValue<"Python", "python",
    "Select python as the default scripting language.">,
  EnumValue<"Default", "default",
    "Select the lldb default as the default scripting language.">,
]>;

def dwim_print_verbosities : EnumDef<"eDWIMPrintVerbosity", [
  EnumValue<"None", "none",
    "Use no verbosity when running dwim-print.">,
  EnumValue<"Expression", "expression",
    "Use partial verbosity when running dwim-print - display a message when "
    "`expression` evaluation is used.">,
  EnumValue<"Full", "full",
    "Use full verbosity when running dwim-print.">,
]>;

def stop_show_column_values : EnumDef<"eStopShowColumn", [
  EnumValue<"AnsiOrCaret", "ansi-or-caret",
    "Highlight the stop column with ANSI terminal codes when color/ANSI mode "
    "is enabled; otherwise, fall back to using a text-only caret (^) as if "
    "\"caret-only\" mode was selected.">,
  EnumValue<"Ansi", "ansi",
    "Highlight the stop column with ANSI terminal codes when running LLDB with "
    "color/ANSI enabled.">,
  EnumValue<"Caret", "caret",
    "Highlight the stop column with a caret character (^) underneath the stop "
    "column. This method introduces a new line in source listings that display "
    "thread stop locations.">,
  EnumValue<"None", "none", "Do not highlight the stop column.">,
]>;

def show_autosuggestion_enum_values : EnumDef<"eAutosuggestion", [
  EnumValue<"Off", "false", "Do not show any autosuggestion.">,
  EnumValue<"On", "true",
    "Show a suggestion sourced from previously entered commands.">,
  EnumValue<"TabMode", "tab-mode",
    "Show the prefix that tab completion would insert for the current line.">,
]>;
