include "llvm/Option/OptParser.td"

class F<string name, string help>
    : Flag<["-", "--"], name>, HelpText<help>;

multiclass B<string name, string help> {
  def NAME : F<name, help>;
  def NAME #_EQ : Joined<["-", "--"], name #"=">,
                   MetaVarName<"<true|false>">,
                   Flags<[HelpHidden]>;
}

multiclass BH<string name, string help> {
  def NAME : F<name, help>, Flags<[HelpHidden]>;
  def NAME #_EQ : Joined<["-", "--"], name #"=">,
                   MetaVarName<"<true|false>">,
                   Flags<[HelpHidden]>;
}

multiclass Eq<string name, string metavar, string help> {
  def NAME #_EQ : Joined<["-", "--"], name #"=">,
                  HelpText<help>, MetaVarName<metavar>;
  def : Separate<["-", "--"], name>, Alias<!cast<Joined>(NAME #_EQ)>;
}

def help : F<"help", "Display available options">;
def help_hidden : F<"help-hidden",
                    "Display all available options, including hidden options">;
def version : F<"version", "Display the version of this program">;
def h : F<"h", "Alias for --help">, Alias<help>, Flags<[HelpHidden]>;

defm offset : Eq<"offset", "<uint>",
                 "Format a range starting at this byte offset.\n"
                 "Multiple ranges can be formatted by specifying\n"
                 "several -offset and -length pairs.\n"
                 "Can only be used with one input file.">;
defm length : Eq<"length", "<uint>",
                 "Format a range of this length (in bytes).\n"
                 "Multiple ranges can be formatted by specifying\n"
                 "several -offset and -length pairs.\n"
                 "When only a single -offset is specified without\n"
                 "-length, clang-format will format up to the end\n"
                 "of the file.\n"
                 "Can only be used with one input file.">;
defm lines : Eq<"lines", "<start line>:<end line>",
                "Format a range of lines (both 1-based).\n"
                "Multiple ranges can be formatted by specifying\n"
                "several -lines arguments.\n"
                "Can't be used with -offset and -length.\n"
                "Can only be used with one input file.">;
defm style : Eq<"style", "<string>",
                "Set coding style. <string> can be:\n"
                "1. A preset: LLVM, GNU, Google, Chromium, Microsoft,\n"
                "   Mozilla, WebKit.\n"
                "2. 'file' to load style configuration from a\n"
                "   .clang-format file in one of the parent directories\n"
                "   of the source file (for stdin, see --assume-filename).\n"
                "   If no .clang-format file is found, falls back to\n"
                "   --fallback-style.\n"
                "   --style=file is the default.\n"
                "3. 'file:<format_file_path>' to explicitly specify\n"
                "   the configuration file.\n"
                "4. \"{key: value, ...}\" to set specific parameters, e.g.:\n"
                "   --style=\"{BasedOnStyle: llvm, IndentWidth: 8}\"">;
defm fallback_style : Eq<"fallback-style", "<string>",
                         "The name of the predefined style used as a\n"
                         "fallback in case clang-format is invoked with\n"
                         "-style=file, but can not find the .clang-format\n"
                         "file to use. Defaults to 'LLVM'.\n"
                         "Use -fallback-style=none to skip formatting.">;
defm assume_filename : Eq<"assume-filename", "<string>",
                         "Set filename used to determine the language and to "
                         "find\n"
                         ".clang-format file.\n"
                         "Only used when reading from stdin.\n"
                         "If this is not passed, the .clang-format file is "
                         "searched\n"
                         "relative to the current working directory when "
                         "reading stdin.\n"
                         "Unrecognized filenames are treated as C++.\n"
                         "supported:\n"
                         "  CSharp: .cs\n"
                         "  Java: .java\n"
                         "  JavaScript: .js .mjs .cjs .ts\n"
                         "  JSON: .json .ipynb\n"
                         "  Objective-C: .m .mm\n"
                         "  Proto: .proto .protodevel\n"
                         "  TableGen: .td\n"
                         "  TextProto: .txtpb .textpb .pb.txt .textproto "
                         ".asciipb\n"
                         "  Verilog: .sv .svh .v .vh">;

defm i : B<"i", "Inplace edit <file>s, if specified.">;
defm output_replacements_xml
    : B<"output-replacements-xml", "Output replacements as XML.">;
defm dump_config : B<"dump-config",
                     "Dump configuration options to stdout and exit.\n"
                     "Can be used with -style option.">;
defm cursor : Eq<"cursor", "<uint>",
                 "The position of the cursor when invoking\n"
                 "clang-format from an editor integration">;
defm sort_includes
    : B<"sort-includes",
        "If set, overrides the include sorting behavior\n"
        "determined by the SortIncludes style flag">;
defm qualifier_alignment
    : Eq<"qualifier-alignment", "<string>",
         "If set, overrides the qualifier alignment style\n"
         "determined by the QualifierAlignment style flag">;
defm files : Eq<"files", "<filename>",
                "A file containing a list of files to process, one per line.">;
defm verbose
    : B<"verbose", "If set, shows the list of processed files">;
defm dry_run
    : B<"dry-run", "If set, do not actually make the formatting changes">;
def n : F<"n", "Alias for --dry-run">, Alias<dry_run>;

defm Wclang_format_violations
    : BH<"Wclang-format-violations",
         "Warnings about individual formatting changes needed. Used only with "
         "--dry-run or -n">;
defm Wno_clang_format_violations
    : BH<"Wno-clang-format-violations",
         "Do not warn about individual formatting changes needed. Used only "
         "with --dry-run or -n">;
defm ferror_limit
    : Eq<"ferror-limit", "<uint>",
         "Set the maximum number of clang-format errors to emit\n"
         "before stopping (0 = no limit).\n"
         "Used only with --dry-run or -n">;
defm Werror
    : B<"Werror", "If set, changes formatting warnings to errors">;
defm Wno_error
    : Eq<"Wno-error", "<warning>",
         "If set, don't error out on the specified warning type.\n"
         "Supported warnings:\n"
         "  unknown - unknown format options are only warned about.\n"
         "This can be used to enable formatting, even if the\n"
         "configuration contains unknown (newer) options.\n"
         "Use with caution, as this might lead to dramatically\n"
         "differing format depending on an option being\n"
         "supported or not.">;
defm fcolor_diagnostics
    : BH<"fcolor-diagnostics",
         "If set, and on a color-capable terminal controls whether or not to "
         "print diagnostics in color">;
defm fno_color_diagnostics
    : BH<"fno-color-diagnostics",
         "If set, and on a color-capable terminal controls whether or not to "
         "print diagnostics in color">;
defm fail_on_incomplete_format
    : B<"fail-on-incomplete-format",
        "If set, fail with exit code 1 on incomplete format.">;
defm list_ignored : BH<"list-ignored", "List ignored files.">;
