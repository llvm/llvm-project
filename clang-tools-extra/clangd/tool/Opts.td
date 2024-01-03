include "llvm/Option/OptParser.td"

class F<string letter, string help> : Flag<["-"], letter>, HelpText<help>;
class FF<string name, string help> : Flag<["-", "--"], name>, HelpText<help>;

multiclass Eq<string name, string help> {
  def NAME #_EQ : Joined<["-", "--"], name #"=">, HelpText<help>;
  def : Separate<["-", "--"], name>, Alias<!cast<Joined>(NAME #_EQ)>;
}

multiclass B<string name, string help1, string help2> {
  def NAME: Flag<["-" ,"--"], name>, HelpText<help1>;
  def no_# NAME: Flag<["-", "--"], "no-" # name>, HelpText<help2>;
}

def grp_generic: OptionGroup<"Generic options">,
                 HelpText<"Generic Options">;

def grp_compile_commands: OptionGroup<"Compile commands">,
                          HelpText<"clangd compilation flags options">;

def grp_features: OptionGroup<"Features">,
                  HelpText<"clangd feature options">;

def grp_misc: OptionGroup<"Misc">,
              HelpText<"clangd miscellaneaous options">;

def grp_protocol: OptionGroup<"Protocol">,
                  HelpText<"clangd protocol and logging options">;

def grp_retired: OptionGroup<"Retired">,
                 HelpText<"clangd flags no longer in use">;

def grp_compat_alias: OptionGroup<"Compatibility Aliases">,
                      HelpText<"compatibility aliases">;

multiclass Retired<string name> {
  def : Joined<["-", "--"], name#"=">, HelpText<"Obsolete flag, ignored">, Group<grp_retired>, Flags<[HelpHidden]>;
  def : Flag<["-", "--"], name>, HelpText<"Obsolete flag, ignored">, Group<grp_retired>, Flags<[HelpHidden]>;
} 

def help : FF<"help", "Display available options (--help-hidden for more)">, Group<grp_generic>;
def help_hidden : FF<"help-hidden", "Display hidden options">, Group<grp_generic>, Flags<[HelpHidden]>;
def : F<"h", "Alias for --help">, Alias<help>, Group<grp_generic>;
def version : FF<"version", "Display the version of this program">, Group<grp_generic>;
def : F<"v", "Alias for --version">, Alias<version>, Group<grp_generic>;

defm compile_args_from :
  Eq<"compile_args_from", "The source of compile commands."
     "Supported values:\n"
     " lsp: All compile commands come from LSP and 'compile_commands.json' files are ignored\n"
     " filesystem: All compile commands come from the 'compile_commands.json' files">,
  Group<grp_compile_commands>,
  MetaVarName<"<value>">,
  Values<"lsp,filesystem">,
  Flags<[HelpHidden]>;

defm compile_commands_dir :
  Eq<"compile-commands-dir",
         "Specify a path to look for compile_commands.json. If path "
         "is invalid, clangd will look in the current directory and "
         "parent paths of each source file">,
  MetaVarName<"<string>">,
  Group<grp_compile_commands>;

defm resource_dir :
  Eq<"resource-dir", "Directory for system clang headers">,
  Group<grp_compile_commands>,
  MetaVarName<"<string>">,
  Flags<[HelpHidden]>;

def query_driver_EQ :
  CommaJoined<["-", "--"], "query-driver=">, 
  HelpText<"Comma separated list of globs for white-listing gcc-compatible "
        "drivers that are safe to execute. Drivers matching any of these globs "
        "will be used to extract system includes. e.g. "
        "/usr/bin/**/clang-*,/path/to/repo/**/g++-*">,
  MetaVarName<"<string>">,
  Group<grp_compile_commands>;

defm all_scopes_completion :
  B<"all-scopes-completion", "Code completion will include index symbols that are "
         "not defined in the scopes (e.g. namespaces) visible from the code completion point. "
         "Such completions can insert scope qualifiers", 
         "Code completion won't include index symbols that are "
         "not defined in the scopes visible from the code completion point.">,
  Group<grp_features>;

def debug_origin :
  FF<"debug-origin", "Show origins of completion items">,
  Group<grp_features>,
  Flags<[HelpHidden]>;

defm background_index :
  B<"background-index","Index project code in the background and persist index on disk.",
                       "Don't index project code in the backgroud.">,
  Group<grp_features>;

defm background_index_priority :
  Eq<"background-index-priority", "Thread priority for building the background index. "
        "The effect of this flag is OS-specific.\n"
        "Supported Values:\n"
        "  background: Minimum priority, runs on idle CPUs. May leave 'performance' cores unused.\n"
        "  low: Reduced priority compared to interactive work.\n"
        "  normal: Reduced priority compared to interactive work." >,
  Values<"background,low,normal">,
  Group<grp_features>;

defm clang_tidy :
  B<"clang-tidy", "Enable clang-tidy diagnostics, enabled by default.", "Disable clang-tidy diagnostics.">,
  Group<grp_features>;

defm completion_parse :
  Eq<"completion-parse", 
     "Whether the clang-parser is used for code-completion\n"
     "Supported Values:\n"
     "  always: Block until the parser can be used\n"
     "  auto: Use text-based completion if the parser is not ready\n"
     "  never: Always used text-based completion">,
  Values<"always,auto,never">,
  Group<grp_features>,
  Flags<[HelpHidden]>;

defm ranking_model :
  Eq<"ranking-model", 
     "Model to use to rank code-completion items.\n"
     "Supported values:\n"
     "  heuristics: Use heuristics to rank code completion items\n"
     "  decision_forest: Use Decision Forest model to rank completion items">,
  Values<"heuristics,decision_forest">,
  Group<grp_features>,
  Flags<[HelpHidden]>;

defm completion_style :
  Eq<"completion-style", 
     "Granularity of code completion suggestions\n"
     "Supported values:\n"
     "  detailed: One completion item for each semantically distinct completion, with full type information\n"
     "  bundled: Similar completion items (e.g. function overloads) are combined. Type information shown where possible">,
  Values<"detailed,bundled">,
  Group<grp_features>;

defm fallback_style :
  Eq<"fallback-style", "clang-format style to apply by default when no .clang-format file is found">,
  Group<grp_features>;

defm function_arg_placeholders :
  B<"function-arg-placeholders", "Completions also contain placeholders for "
  "method parameters (enabled by default)", 
  "Completions contain only parentheses for function calls.">,
  Group<grp_features>;

defm header_insertion :
  Eq<"header-insertion", 
     "Add #include directives when accepting code completions\n"
     "Supported values:\n"
     "  iwyu: Include what you use. Insert the owning header for top-level symbols, unless the"
          " header is already directly included or the symbol is forward-declared\n"
     "  never: Never insert #include directives as part of code completion">,
  Values<"iwyu,never">,
  Group<grp_features>;

defm import_insertions :
  B<"import-insertions", "Add #import directives when accepting code completions "
  "or fixing includes in Objective-C code (enabled by default)", 
  "Do not add #import directives when accepting code completions or fixing includes "
  "in Objective-C code">,
  Group<grp_features>;

defm header_insertion_decorators :
  B<"header-insertion-decorators", "Prepend a circular dot or space before the completion "
    "label, depending on whether an include line will be inserted or not (enabled by default).",
    "Do not prepend a circular dot or space before the completion label.">,
  Group<grp_features>;

def hidden_features :
  FF<"hidden-features", "Enable hidden features mostly useful to clangd developers">,
  Group<grp_features>,
  Flags<[HelpHidden]>;

def include_ineligible_results :
  FF<"include-ineligible-results", "Include ineligible completion results (e.g. private members)">,
  Group<grp_features>,
  Flags<[HelpHidden]>;

defm limit_results :
  Eq<"limit-results", "Limit the number of results returned by clangd. 0 means no limit (default=100)">,
  Group<grp_features>;

defm limit_references :
  Eq<"limit-references", "Limit the number of references returned by clangd. 0 means no limit (default=1000)">,
  Group<grp_features>;

defm rename_file_limit :
  Eq<"rename-file-limit", "Limit the number of files to be affected by symbol renaming. 0 means no limit (default=50)">,
  Group<grp_features>;

def tweaks_EQ :
  CommaJoined<["-", "--"], "tweaks=">, 
  HelpText<"Specify a list of Tweaks to enable (only for clangd developers).">,
  Group<grp_features>,
  Flags<[HelpHidden]>;

defm j :
  Eq<"j", "Number of async workers used by clangd. Background index also uses this many workers.">,
  Group<grp_misc>;

defm index_file :
  Eq<"index-file", "Index file to build the static index. The file must have been created "
        "by a compatible clangd-indexer\n"
        "WARNING: This option is experimental only, and will be removed "
        "eventually. Don't rely on it">,
  Group<grp_misc>,
  Flags<[HelpHidden]>;

def lit_test :
   FF<"lit-test", "Abbreviation for -input-style=delimited -pretty -sync "
         "-enable-test-scheme -enable-config=0 -log=verbose -crash-pragmas. "
         "Also sets config options: Index.StandardLibrary=false. "
         "Intended to simplify lit tests">,
  Group<grp_misc>,
  Flags<[HelpHidden]>;

def crash_pragmas :
  FF<"crash-pragmas", "Respect `#pragma clang __debug crash` and friends.">,
  Group<grp_misc>,
  Flags<[HelpHidden]>;

defm check : Eq<"check", "Parse one file in isolation instead of acting as a language server. "
         "Useful to investigate/reproduce crashes or configuration problems. "
         "With --check=<filename>, attempts to parse a particular file.">,
  Group<grp_misc>;

defm pch_storage : Eq<"pch-storage", 
         "Storing PCHs in memory increases memory usages, but may improve performance\n"
         "Supported values:\n"
         "  disk: store PCHs on disk\n"
         "  memory: store PCHs in memory">,
  MetaVarName<"<value>">,
  Group<grp_misc>;

defm sync : 
	B<"sync", "Handle client requests on main thread. Background index still uses its own thread.",
						"Do not handle client requests on main thread.">,
   Group<grp_misc>,
   Flags<[HelpHidden]>;

defm input_style :
  Eq<"input-style", 
  "Input JSON stream encoding.\n"
  "Supported values:\n"
  "  standard: usual LSP protocol\n"
  "  delimited: messages delimited by --- lines, with # comment support">,
  Values<"standard,delimited">,
  MetaVarName<"<value>">,
  Group<grp_protocol>,
  Flags<[HelpHidden]>;

def enable_test_uri_scheme :
  FF<"enable-test-uri-scheme", "Enable 'test:' URI scheme. Only use in lit tests">,
  Group<grp_protocol>,
  Flags<[HelpHidden]>;

defm path_mappings : Eq<"path-mappings", "Translates between client paths (as seen by a remote editor) and "
        "server paths (where clangd sees files on disk). "
        "Comma separated list of '<client_path>=<server_path>' pairs, the "
        "first entry matching a given path is used. "
        "e.g. /home/project/incl=/opt/include,/home/project=/workarea/project">,
  Group<grp_protocol>;

defm input_mirror_file :
  Eq<"input-mirror-file", "Mirror all LSP input to the specified file. Useful for debugging">,
  Group<grp_protocol>,
  Flags<[HelpHidden]>;

defm log : 
  Eq<"log", 
    "Verbosity of log messages written to stderr\n" 
    "Supported values:\n"
    "  error: Error messages only\n"
    "  info: High level execution tracing\n"
    "  verbose: Low level details">,
  Values<"error,info,verbose">,
  MetaVarName<"<value>">,
  Group<grp_protocol>;

defm offset_encoding : 
  Eq<"offset-encoding", 
     "Force the offsetEncoding used for character positions. This bypasses negotiation via client capabilities\n"
     "Supported values:\n"
     "  utf-8: Offsets are in UTF-8 bytes\n"
     "  utf-16: Offsets are in UTF-16 code units\n"
     "  utf-32: Offsets are in unicode codepoints">,
  Values<"utf-8,utf-16,utf-32">,
  MetaVarName<"<value>">,
  Group<grp_protocol>;

def pretty : FF<"pretty", "Pretty-print JSON output">, Group<grp_protocol>;

defm enable_config : 
  B<"enable-config",
    "Read user and project configuration from YAML files.\n"
        "Project config is from a .clangd file in the project directory.\n"
        "User config is from clangd/config.yaml in the following directories:\n"
        "  Windows: %USERPROFILE%\\AppData\\Local\n"
        "  Mac OS: ~/Library/Preferences/\n"
        "  Others: $XDG_CONFIG_HOME, usually ~/.config\n"
        "Configuration is documented at https://clangd.llvm.org/config.html", 
    "Do not read user and project configuration form YAML files.">,
  Group<grp_misc>;

defm use_dirty_headers :
  B<"use-dirty-headers",
  "Use files open in the editor when parsing headers instead of reading from the disk",
  "Do not Use files open in the editor when parsing headers instead of reading from the disk">,
  Group<grp_misc>,
  Flags<[HelpHidden]>;

defm parse_forwarding_functions :
  B<"parse-forwarding-functions", 
  "Parse all emplace-like functions in included headers.",
  "Do not parse all emplace-like functions in included headers.">,
  Group<grp_misc>,
  Flags<[HelpHidden]>;

def MallocTrimVis: OptionVisibility;

defm malloc_trim :
  B<"malloc-trim",
    "Release memory periodically via malloc_trim(3) enabled by default.",
    "Do not use malloc_trim(3) to release memory periodically.">,
  Group<grp_misc>,
  Visibility<[MallocTrimVis]>;

def RemoteVis: OptionVisibility;

defm remote_index_address :
  Eq<"remote-index-address", "Address of the remote index server">,
  Group<grp_features>,
  Visibility<[RemoteVis]>;

defm project_root :
  Eq<"project-root", "Path to the project root. Requires remote-index-address to be set.">,
  Group<grp_features>,
  Visibility<[RemoteVis]>;

defm : Retired<"index">;
defm : Retired<"suggest-missing-includes">;
defm : Retired<"recovery-ast">;
defm : Retired<"recovery-ast-type">;
defm : Retired<"async-preamble">;
defm : Retired<"collect-main-file-refs">;
defm : Retired<"cross-file-rename">;
defm : Retired<"clang-tidy-checks">;
defm : Retired<"inlay-hints">;
defm : Retired<"folding-ranges">;
defm : Retired<"include-cleaner-stdlib">;

// Compatibility aliases
def : Flag<["--", "-"], "all-scopes-completion=true">, Alias<all_scopes_completion>, HelpText<"Alias for --all-scopes-completion">, Flags<[HelpHidden]>, Group<grp_compat_alias>;
def : Flag<["--", "-"], "all-scopes-completion=false">, Alias<no_all_scopes_completion>, HelpText<"Alias for --no-all-scopes-completion">, Flags<[HelpHidden]>, Group<grp_compat_alias>;
def : Flag<["--", "-"], "all-scopes-completion=0">, Alias<no_all_scopes_completion>, HelpText<"Alias for --no-all-scopes-completion">, Flags<[HelpHidden]>, Group<grp_compat_alias>;

def : Flag<["--", "-"], "background-index=true">, Alias<background_index>, HelpText<"Alias for --background-index">, Flags<[HelpHidden]>, Group<grp_compat_alias>;
def : Flag<["--", "-"], "background-index=false">, Alias<no_background_index>, HelpText<"Alias for --no-background-index">, Flags<[HelpHidden]>, Group<grp_compat_alias>;
def : Flag<["--", "-"], "background-index=0">, Alias<no_background_index>, HelpText<"Alias for --no-background-index">, Flags<[HelpHidden]>, Group<grp_compat_alias>;

def : Flag<["--", "-"], "clang-tidy=true">, Alias<clang_tidy>, HelpText<"Alias for --clang-tidy">, Flags<[HelpHidden]>, Group<grp_compat_alias>;
def : Flag<["--", "-"], "clang-tidy=false">, Alias<no_clang_tidy>, HelpText<"Alias for --no-clang-tidy">, Flags<[HelpHidden]>, Group<grp_compat_alias>;
def : Flag<["--", "-"], "clang-tidy=0">, Alias<no_clang_tidy>, HelpText<"Alias for --no-clang-tidy">, Flags<[HelpHidden]>, Group<grp_compat_alias>;

def : Flag<["--", "-"], "function-arg-placeholders=true">, Alias<function_arg_placeholders>, HelpText<"Alias for --function-arg-placeholders">, Flags<[HelpHidden]>, Group<grp_compat_alias>;
def : Flag<["--", "-"], "function-arg-placeholders=false">, Alias<no_function_arg_placeholders>, HelpText<"Alias for --no-function-arg-placeholders">, Flags<[HelpHidden]>, Group<grp_compat_alias>;
def : Flag<["--", "-"], "function-arg-placeholders=0">, Alias<no_function_arg_placeholders>, HelpText<"Alias for --no-function-arg-placeholders">, Flags<[HelpHidden]>, Group<grp_compat_alias>;

def : Flag<["--", "-"], "enable-config=true">, Alias<enable_config>, HelpText<"Alias for --enable-config">, Flags<[HelpHidden]>, Group<grp_compat_alias>;
def : Flag<["--", "-"], "enable-config=false">, Alias<no_enable_config>, HelpText<"Alias for --no-enable-config">, Flags<[HelpHidden]>, Group<grp_compat_alias>;
def : Flag<["--", "-"], "enable-config=0">, Alias<no_enable_config>, HelpText<"Alias for --no-enable-config">, Flags<[HelpHidden]>, Group<grp_compat_alias>;

def : Flag<["--", "-"], "use-dirty-headers=true">, Alias<use_dirty_headers>, HelpText<"Alias for --use-dirty-headers">, Flags<[HelpHidden]>, Group<grp_compat_alias>;
def : Flag<["--", "-"], "use-dirty-headers=false">, Alias<no_use_dirty_headers>, HelpText<"Alias for --no-use-dirty-headers">, Flags<[HelpHidden]>, Group<grp_compat_alias>;
def : Flag<["--", "-"], "use-dirty-headers=0">, Alias<no_use_dirty_headers>, HelpText<"Alias for --no-use-dirty-headers">, Flags<[HelpHidden]>, Group<grp_compat_alias>;

def : Flag<["--", "-"], "sync=true">, Alias<sync>, HelpText<"Alias for --sync">, Flags<[HelpHidden]>, Group<grp_compat_alias>;
def : Flag<["--", "-"], "sync=false">, Alias<no_sync>, HelpText<"Alias for --no-sync">, Flags<[HelpHidden]>, Group<grp_compat_alias>;
def : Flag<["--", "-"], "sync=0">, Alias<no_sync>, HelpText<"Alias for --no-sync">, Flags<[HelpHidden]>, Group<grp_compat_alias>;

//Check flags
def grp_check:
  OptionGroup<"Check">,
  HelpText<"clangd check specific options. Used for test or diagnostics">,
  Flags<[HelpHidden]>;

defm check_tidy_time :
	Eq<"check-tidy-time",
		 "Print the overhead of checks matching this glob">,
  Group<grp_check>,
	Flags<[HelpHidden]>;

defm check_file_lines:
	Eq<"check-lines",
		 "Limits the range of tokens in -check file on which "
     "various features are tested. Example --check-lines=3-7 restricts "
     "testing to lines 3 to 7 (inclusive) or --check-lines=5 to restrict "
     "to one line. Default is testing entire file.">,
  Group<grp_check>,
	Flags<[HelpHidden]>;

defm check_locations :
	B<"check-locations",
	  "Runs certain features (e.g. hover) at each point in the file. Somewhat slow. (enable by default)",
		"Do not run certain features (e.g. hover) at each point in the file.">,
  Group<grp_check>,
	Flags<[HelpHidden]>;

def check_completion :
	FF<"check-completion",
		 "Run code-completion at each point (slow)">,
  Group<grp_check>,
	Flags<[HelpHidden]>;

def check_warnings :
	FF<"check-warnings",
		 "Print warnings as well as errors.">,
  Group<grp_check>,
	Flags<[HelpHidden]>;
