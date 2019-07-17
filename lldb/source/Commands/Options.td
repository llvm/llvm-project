include "OptionsBase.td"

let Command = "target modules dump symtab" in {
  def tm_sort : Option<"sort", "s">, Group<1>,
    Desc<"Supply a sort order when dumping the symbol table.">,
    EnumArg<"SortOrder", "OptionEnumValues(g_sort_option_enumeration)">;
}

let Command = "help" in {
  def help_hide_aliases : Option<"hide-aliases", "a">,
    Desc<"Hide aliases in the command list.">;
  def help_hide_user : Option<"hide-user-commands", "u">,
    Desc<"Hide user-defined commands from the list.">;
  def help_show_hidden : Option<"show-hidden-commands", "h">,
    Desc<"Include commands prefixed with an underscore.">;
}

let Command = "settings set" in {
  def setset_global : Option<"global", "g">, Arg<"Filename">,
    Completion<"DiskFile">,
    Desc<"Apply the new value to the global default value.">;
  def setset_force : Option<"force", "f">,
    Desc<"Force an empty value to be accepted as the default.">;
}

let Command = "settings write" in {
  def setwrite_file : Option<"file", "f">, Required, Arg<"Filename">,
    Completion<"DiskFile">,
    Desc<"The file into which to write the settings.">;
  def setwrite_append : Option<"append", "a">,
    Desc<"Append to saved settings file if it exists.">;
}

let Command = "settings read" in {
  def setread_file : Option<"file", "f">, Required, Arg<"Filename">,
    Completion<"DiskFile">,
    Desc<"The file from which to read the settings.">;
}

let Command = "breakpoint list" in {
  def blist_internal : Option<"internal", "i">,
    Desc<"Show debugger internal breakpoints">;
  def blist_brief : Option<"brief", "b">, Group<1>,
    Desc<"Give a brief description of the breakpoint (no location info).">;
  def blist_full : Option<"full", "f">, Group<2>,
    Desc<"Give a full description of the breakpoint and its locations.">;
  def blist_verbose : Option<"verbose", "v">, Group<3>,
    Desc<"Explain everything we know about the breakpoint (for debugging "
    "debugger bugs).">;
  def blist_dummy_bp : Option<"dummy-breakpoints", "D">,
    Desc<"List Dummy breakpoints - i.e. breakpoints set before a file is "
    "provided, which prime new targets.">;
}

let Command = "watchpoint list" in {
  def watchpoint_list_brief : Option<"brief", "b">, Group<1>, Desc<"Give a "
    "brief description of the watchpoint (no location info).">;
  def watchpoint_list_full : Option<"full", "f">, Group<2>, Desc<"Give a full "
    "description of the watchpoint and its locations.">;
  def watchpoint_list_verbose : Option<"verbose", "v">, Group<3>, Desc<"Explain"
    "everything we know about the watchpoint (for debugging debugger bugs).">;
}

let Command = "watchpoint ignore" in {
  def watchpoint_ignore_ignore_count : Option<"ignore-count", "i">,
    Arg<"Count">, Required, Desc<"Set the number of times this watchpoint is"
    " skipped before stopping.">;
}

let Command = "watchpoint modify" in {
  def watchpoint_modify_condition : Option<"condition", "c">, Arg<"Expression">,
    Desc<"The watchpoint stops only if this condition expression evaluates "
    "to true.">;
}

let Command = "watchpoint command add" in {
  def watchpoint_command_add_one_liner : Option<"one-liner", "o">, Group<1>,
    Arg<"OneLiner">, Desc<"Specify a one-line watchpoint command inline. Be "
    "sure to surround it with quotes.">;
  def watchpoint_command_add_stop_on_error : Option<"stop-on-error", "e">,
    Arg<"Boolean">, Desc<"Specify whether watchpoint command execution should "
    "terminate on error.">;
  def watchpoint_command_add_script_type : Option<"script-type", "s">,
    EnumArg<"None", "ScriptOptionEnum()">, Desc<"Specify the language for the"
    " commands - if none is specified, the lldb command interpreter will be "
    "used.">;
  def watchpoint_command_add_python_function : Option<"python-function", "F">,
    Group<2>, Arg<"PythonFunction">, Desc<"Give the name of a Python function "
    "to run as command for this watchpoint. Be sure to give a module name if "
    "appropriate.">;
}
