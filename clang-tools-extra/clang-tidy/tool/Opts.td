include "llvm/Option/OptParser.td"

def clang_tidy_group : OptionGroup<"clang-tidy">,
                       HelpText<"clang-tidy options">;
def generic_group : OptionGroup<"generic">, HelpText<"Generic Options">;

class F<string name, string help>
    : Flag<["--", "-"], name>, HelpText<help>, Group<clang_tidy_group>;

multiclass B<string name, string help> {
  def NAME : F<name, help>;
  def NAME #_EQ : Joined<["--", "-"], name #"=">,
                   MetaVarName<"<true|false>">,
                   Group<clang_tidy_group>, Flags<[HelpHidden]>;
}

multiclass BH<string name> {
  def NAME : Flag<["--", "-"], name>, Group<clang_tidy_group>,
             Flags<[HelpHidden]>;
  def NAME #_EQ : Joined<["--", "-"], name #"=">,
                   MetaVarName<"<true|false>">,
                   Group<clang_tidy_group>, Flags<[HelpHidden]>;
}

multiclass Eq<string name, string metavar, string help> {
  def NAME #_EQ : Joined<["--", "-"], name #"=">,
                  HelpText<help>, MetaVarName<metavar>,
                  Group<clang_tidy_group>;
  def : Separate<["--", "-"], name>, Alias<!cast<Joined>(NAME #_EQ)>;
}

class GF<string name, string help>
    : Flag<["--", "-"], name>, HelpText<help>, Group<generic_group>;

def help : GF<"help", "Display available options (--help-hidden for more)">;
def help_hidden : GF<"help-hidden",
                     "Display all available options, including hidden options">,
                  Flags<[HelpHidden]>;
def help_list
    : GF<"help-list",
         "Display list of available options (--help-list-hidden for more)">,
      Alias<help>;
def help_list_hidden
    : GF<"help-list-hidden", "Display list of all available options">,
      Alias<help_hidden>, Flags<[HelpHidden]>;
def version : GF<"version", "Display the version of this program">;
def h : GF<"h", "Alias for --help">, Alias<help>, Flags<[HelpHidden]>;

defm checks : Eq<"checks", "<string>",
                 "Comma-separated list of globs with optional '-'\n"
                 "prefix. Globs are processed in order of\n"
                 "appearance in the list. Globs without '-'\n"
                 "prefix add checks with matching names to the\n"
                 "set, globs with the '-' prefix remove checks\n"
                 "with matching names from the set of enabled\n"
                 "checks. This option's value is appended to the\n"
                 "value of the 'Checks' option in .clang-tidy\n"
                 "file, if any.">;
defm warnings_as_errors
    : Eq<"warnings-as-errors", "<string>",
         "Upgrades warnings to errors. Same format as\n"
         "'-checks'.\n"
         "This option's value is appended to the value of\n"
         "the 'WarningsAsErrors' option in .clang-tidy\n"
         "file, if any.">;
defm header_filter
    : Eq<"header-filter", "<string>",
         "Regular expression matching the names of the\n"
         "headers to output diagnostics from. The default\n"
         "value is '.*', i.e. diagnostics from all non-system\n"
         "headers are displayed by default. Diagnostics\n"
         "from the main file of each translation unit are\n"
         "always displayed.\n"
         "Can be used together with -line-filter.\n"
         "This option overrides the 'HeaderFilterRegex'\n"
         "option in .clang-tidy file, if any.">;
defm exclude_header_filter
    : Eq<"exclude-header-filter", "<string>",
         "Regular expression matching the names of the\n"
         "headers to exclude diagnostics from. Diagnostics\n"
         "from the main file of each translation unit are\n"
         "always displayed.\n"
         "Must be used together with --header-filter.\n"
         "Can be used together with -line-filter.\n"
         "This option overrides the 'ExcludeHeaderFilterRegex'\n"
         "option in .clang-tidy file, if any.">;
defm system_headers
    : B<"system-headers",
        "Display the errors from system headers.\n"
        "This option overrides the 'SystemHeaders' option\n"
        "in .clang-tidy file, if any.">;
defm line_filter
    : Eq<"line-filter", "<string>",
         "List of files and line ranges to output diagnostics from.\n"
         "The range is inclusive on both ends. Can be used together\n"
         "with -header-filter. The format of the list is a JSON\n"
         "array of objects. For example:\n"
         "  [\n"
         "    {\"name\":\"file1.cpp\",\"lines\":[[1,3],[5,7]]},\n"
         "    {\"name\":\"file2.h\"}\n"
         "  ]\n"
         "This will output diagnostics from 'file1.cpp' only for\n"
         "the line ranges [1,3] and [5,7], as well as all from the\n"
         "entire 'file2.h'.">;
defm fix
    : B<"fix", "Apply suggested fixes. Without -fix-errors\n"
               "clang-tidy will bail out if any compilation\n"
               "errors were found.">;
defm fix_errors
    : B<"fix-errors",
        "Apply suggested fixes even if compilation\n"
        "errors were found. If compiler errors have\n"
        "attached fix-its, clang-tidy will apply them as\n"
        "well.">;
defm fix_notes
    : B<"fix-notes",
        "If a warning has no fix, but a single fix can\n"
        "be found through an associated diagnostic note,\n"
        "apply the fix.\n"
        "Specifying this flag will implicitly enable the\n"
        "'--fix' flag.">;
defm format_style
    : Eq<"format-style", "<string>",
         "Style for formatting code around applied fixes:\n"
         "  - 'none' (default) turns off formatting\n"
         "  - 'file' (literally 'file', not a placeholder)\n"
         "    uses .clang-format file in the closest parent\n"
         "    directory\n"
         "  - '{ <json> }' specifies options inline, e.g.\n"
         "    -format-style='{BasedOnStyle: llvm, IndentWidth: 8}'\n"
         "  - 'llvm', 'google', 'webkit', 'mozilla'\n"
         "See clang-format documentation for the up-to-date\n"
         "information about formatting styles and options.\n"
         "This option overrides the 'FormatStyle` option in\n"
         ".clang-tidy file, if any.">;
defm list_checks
    : B<"list-checks",
        "List all enabled checks and exit. Use with\n"
        "-checks=* to list all available checks.">;
defm explain_config
    : B<"explain-config",
        "For each enabled check explains, where it is\n"
        "enabled, i.e. in clang-tidy binary, command\n"
        "line or a specific configuration file.">;
defm config
    : Eq<"config", "<string>",
         "Specifies a configuration in YAML/JSON format:\n"
         "  -config=\"{Checks: '*',\n"
         "            CheckOptions: {x: y}}\"\n"
         "When the value is empty, clang-tidy will\n"
         "attempt to find a file named .clang-tidy for\n"
         "each source file in its parent directories.">;
defm config_file
    : Eq<"config-file", "<filename>",
         "Specify the path of .clang-tidy or custom config file:\n"
         " e.g. --config-file=/some/path/myTidyConfigFile\n"
         "This option internally works exactly the same way as\n"
         " --config option after reading specified config file.\n"
         "Use either --config-file or --config, not both.">;
defm dump_config
    : B<"dump-config",
        "Dumps configuration in the YAML format to\n"
         "stdout. This option can be used along with a\n"
         "file name (and '--' if the file is outside of a\n"
         "project with configured compilation database).\n"
         "The configuration used for this file will be\n"
         "printed.\n"
         "Use along with -checks=* to include\n"
         "configuration of all checks.">;
defm enable_check_profile
    : B<"enable-check-profile",
        "Enable per-check timing profiles, and print a\n"
        "report to stderr.">;
defm store_check_profile
    : Eq<"store-check-profile", "<prefix>",
         "By default reports are printed in tabulated\n"
         "format to stderr. When this option is passed,\n"
         "these per-TU profiles are instead stored as JSON.">;
defm allow_enabling_analyzer_alpha_checkers
    : BH<"allow-enabling-analyzer-alpha-checkers">;
defm enable_module_headers_parsing
    : B<"enable-module-headers-parsing",
        "Enables preprocessor-level module header parsing\n"
        "for C++20 and above, empowering specific checks\n"
        "to detect macro definitions within modules. This\n"
        "feature may cause performance and parsing issues\n"
        "and is therefore considered experimental.">;
defm export_fixes
    : Eq<"export-fixes", "<filename>",
         "YAML file to store suggested fixes in. The\n"
         "stored fixes can be applied to the input source\n"
         "code with clang-apply-replacements.">;
defm quiet
    : B<"quiet",
        "Run clang-tidy in quiet mode. This suppresses\n"
        "printing statistics about ignored warnings and\n"
        "warnings treated as errors if the respective\n"
        "options are specified.">;
defm vfsoverlay
    : Eq<"vfsoverlay", "<filename>",
         "Overlay the virtual filesystem described by file\n"
         "over the real file system.">;
defm use_color
    : B<"use-color",
        "Use colors in diagnostics. If not set, colors\n"
        "will be used if the terminal connected to\n"
        "standard output supports colors.\n"
        "This option overrides the 'UseColor' option in\n"
        ".clang-tidy file, if any.">;
defm verify_config
    : B<"verify-config",
        "Check the config files to ensure each check and\n"
        "option is recognized without running any checks.">;
defm allow_no_checks
    : B<"allow-no-checks",
        "Allow empty enabled checks. This suppresses\n"
        "the \"no checks enabled\" error when disabling\n"
        "all of the checks.">;
defm experimental_custom_checks
    : B<"experimental-custom-checks",
        "Enable experimental clang-query based\n"
        "custom checks.\n"
        "see https://clang.llvm.org/extra/clang-tidy/"
        "QueryBasedCustomChecks.html.">;
defm removed_arg
    : Eq<"removed-arg", "<arg>",
         "List of arguments to remove from the command\n"
         "line sent to the compiler. Please note that\n"
         "removing arguments might change the semantic\n"
         "of the analyzed code, possibly leading to\n"
         "compiler errors, false positives or\n"
         "false negatives. This option is applied\n"
         "before --extra-arg and --extra-arg-before">;

defm extra_arg
    : Eq<"extra-arg", "<arg>",
         "Additional argument to append to the compiler command line">;
defm extra_arg_before
    : Eq<"extra-arg-before", "<arg>",
         "Additional argument to prepend to the compiler command line">;
def p : Separate<["-", "--"], "p">, HelpText<"Build path">,
        MetaVarName<"<build-path>">, Group<clang_tidy_group>;
def p_EQ : Joined<["-", "--"], "p=">, Alias<p>, Flags<[HelpHidden]>;
defm load
    : Eq<"load", "<pluginfilename>", "Load the specified plugin">;
