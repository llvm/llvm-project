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

let Command = "type summary add" in {
  def type_summary_add_category : Option<"category", "w">, Arg<"Name">, Desc<"Add this to the given category instead of the default one.">;
  def type_summary_add_cascade : Option<"cascade", "C">, Arg<"Boolean">, Desc<"If true, cascade through typedef chains.">;
  def type_summary_add_no_value : Option<"no-value", "v">, Desc<"Don't show the value, just show the summary, for this type.">;
  def type_summary_add_skip_pointers : Option<"skip-pointers", "p">, Desc<"Don't use this format for pointers-to-type objects.">;
  def type_summary_add_skip_references : Option<"skip-references", "r">, Desc<"Don't use this format for references-to-type objects.">;
  def type_summary_add_regex : Option<"regex", "x">, Desc<"Type names are actually regular expressions.">;
  def type_summary_add_inline_children : Option<"inline-children", "c">, Group<1>, Required, Desc<"If true, inline all child values into summary string.">;
  def type_summary_add_omit_names : Option<"omit-names", "O">, Group<1>, Desc<"If true, omit value names in the summary display.">;
  def type_summary_add_summary_string : Option<"summary-string", "s">, Group<2>, Arg<"SummaryString">, Required, Desc<"Summary string used to display text and object contents.">;
  def type_summary_add_python_script : Option<"python-script", "o">, Group<3>, Arg<"PythonScript">, Desc<"Give a one-liner Python script as part of the command.">;
  def type_summary_add_python_function : Option<"python-function", "F">, Group<3>, Arg<"PythonFunction">, Desc<"Give the name of a Python function to use for this type.">;
  def type_summary_add_input_python : Option<"input-python", "P">, Group<3>, Desc<"Input Python code to use for this type manually.">;
  def type_summary_add_expand : Option<"expand", "e">, Groups<[2,3]>, Desc<"Expand aggregate data types to show children on separate lines.">;
  def type_summary_add_hide_empty : Option<"hide-empty", "h">, Groups<[2,3]>, Desc<"Do not expand aggregate data types with no children.">;
  def type_summary_add_name : Option<"name", "n">, Groups<[2,3]>, Arg<"Name">, Desc<"A name for this summary string.">;
}

let Command = "type synth add" in {
  def type_synth_add_cascade : Option<"cascade", "C">, Arg<"Boolean">, Desc<"If true, cascade through typedef chains.">;
  def type_synth_add_skip_pointers : Option<"skip-pointers", "p">, Desc<"Don't use this format for pointers-to-type objects.">;
  def type_synth_add_skip_references : Option<"skip-references", "r">, Desc<"Don't use this format for references-to-type objects.">;
  def type_synth_add_category : Option<"category", "w">, Arg<"Name">, Desc<"Add this to the given category instead of the default one.">;
  def type_synth_add_python_class : Option<"python-class", "l">, Group<2>, Arg<"PythonClass">, Desc<"Use this Python class to produce synthetic children.">;
  def type_synth_add_input_python : Option<"input-python", "P">, Group<3>, Desc<"Type Python code to generate a class that provides synthetic children.">;
  def type_synth_add_regex : Option<"regex", "x">, Desc<"Type names are actually regular expressions.">;
}

let Command = "type format add" in {
  def type_format_add_category : Option<"category", "w">, Arg<"Name">, Desc<"Add this to the given category instead of the default one.">;
  def type_format_add_cascade : Option<"cascade", "C">, Arg<"Boolean">, Desc<"If true, cascade through typedef chains.">;
  def type_format_add_skip_pointers : Option<"skip-pointers", "p">, Desc<"Don't use this format for pointers-to-type objects.">;
  def type_format_add_skip_references : Option<"skip-references", "r">, Desc<"Don't use this format for references-to-type objects.">;
  def type_format_add_regex : Option<"regex", "x">, Desc<"Type names are actually regular expressions.">;
  def type_format_add_type : Option<"type", "t">, Group<2>, Arg<"Name">, Desc<"Format variables as if they were of this type.">;
}

let Command = "type formatter delete" in {
  def type_formatter_delete_all : Option<"all", "a">, Group<1>, Desc<"Delete from every category.">;
  def type_formatter_delete_category : Option<"category", "w">, Group<2>, Arg<"Name">, Desc<"Delete from given category.">;
  def type_formatter_delete_language : Option<"language", "l">, Group<3>, Arg<"Language">, Desc<"Delete from given language's category.">;
}

let Command = "type formatter clear" in {
  def type_formatter_clear_all : Option<"all", "a">, Desc<"Clear every category.">;
}

let Command = "type formatter list" in {
  def type_formatter_list_category_regex : Option<"category-regex", "w">, Group<1>, Arg<"Name">, Desc<"Only show categories matching this filter.">;
  def type_formatter_list_language : Option<"language", "l">, Group<2>, Arg<"Language">, Desc<"Only show the category for a specific language.">;
}

let Command = "type category define" in {
  def type_category_define_enabled : Option<"enabled", "e">, Desc<"If specified, this category will be created enabled.">;
  def type_category_define_language : Option<"language", "l">, Arg<"Language">, Desc<"Specify the language that this category is supported for.">;
}

let Command = "type category enable" in {
  def type_category_enable_language : Option<"language", "l">, Arg<"Language">, Desc<"Enable the category for this language.">;
}

let Command = "type category disable" in {
  def type_category_disable_language : Option<"language", "l">, Arg<"Language">, Desc<"Enable the category for this language.">;
}

let Command = "type filter add" in {
  def type_filter_add_cascade : Option<"cascade", "C">, Arg<"Boolean">, Desc<"If true, cascade through typedef chains.">;
  def type_filter_add_skip_pointers : Option<"skip-pointers", "p">, Desc<"Don't use this format for pointers-to-type objects.">;
  def type_filter_add_skip_references : Option<"skip-references", "r">, Desc<"Don't use this format for references-to-type objects.">;
  def type_filter_add_category : Option<"category", "w">, Arg<"Name">, Desc<"Add this to the given category instead of the default one.">;
  def type_filter_add_child : Option<"child", "c">, Arg<"ExpressionPath">, Desc<"Include this expression path in the synthetic view.">;
  def type_filter_add_regex : Option<"regex", "x">, Desc<"Type names are actually regular expressions.">;
}

let Command = "type lookup" in {
  def type_lookup_show_help : Option<"show-help", "h">, Desc<"Display available help for types">;
  def type_lookup_language : Option<"language", "l">, Arg<"Language">, Desc<"Which language's types should the search scope be">;
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
