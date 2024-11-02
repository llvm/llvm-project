include "llvm/Option/OptParser.td"

// Convenience classes for long options which only accept two dashes. For lld
// specific or newer long options, we prefer two dashes to avoid collision with
// short options. For many others, we have to accept both forms to be compatible
// with GNU ld.
class FF<string name> : Flag<["--"], name>;
class JJ<string name>: Joined<["--"], name>;

multiclass EEq<string name, string help> {
  def NAME: Separate<["--"], name>;
  def NAME # _eq: Joined<["--"], name # "=">, Alias<!cast<Separate>(NAME)>,
    HelpText<help>;
}

multiclass BB<string name, string help1, string help2> {
  def NAME: Flag<["--"], name>, HelpText<help1>;
  def no_ # NAME: Flag<["--"], "no-" # name>, HelpText<help2>;
}

// For options whose names are multiple letters, either one dash or
// two can precede the option name except those that start with 'o'.
class F<string name>: Flag<["--", "-"], name>;
class J<string name>: Joined<["--", "-"], name>;
class S<string name>: Separate<["--", "-"], name>;

multiclass Eq<string name, string help> {
  def NAME: Separate<["--", "-"], name>;
  def NAME # _eq: Joined<["--", "-"], name # "=">, Alias<!cast<Separate>(NAME)>,
    HelpText<help>;
}

multiclass B<string name, string help1, string help2> {
  def NAME: Flag<["--", "-"], name>, HelpText<help1>;
  def no_ # NAME: Flag<["--", "-"], "no-" # name>, HelpText<help2>;
}

// The following flags are shared with the ELF linker
def Bsymbolic: F<"Bsymbolic">, HelpText<"Bind defined symbols locally">;

def Bdynamic: F<"Bdynamic">, HelpText<"Link against shared libraries (default)">;

def Bstatic: F<"Bstatic">, HelpText<"Do not link against shared libraries">;

defm color_diagnostics: B<"color-diagnostics",
  "Alias for --color-diagnostics=always",
  "Alias for --color-diagnostics=never">;
def color_diagnostics_eq: J<"color-diagnostics=">,
  HelpText<"Use colors in diagnostics (default: auto)">,
  MetaVarName<"[auto,always,never]">;

def compress_relocations: F<"compress-relocations">,
  HelpText<"Compress the relocation targets in the code section.">;

defm demangle: B<"demangle",
    "Demangle symbol names",
    "Do not demangle symbol names">;

def emit_relocs: F<"emit-relocs">, HelpText<"Generate relocations in output">;

def error_unresolved_symbols: F<"error-unresolved-symbols">,
  HelpText<"Report unresolved symbols as errors">;

defm export_dynamic: B<"export-dynamic",
    "Put symbols in the dynamic symbol table",
    "Do not put symbols in the dynamic symbol table (default)">;

def entry: S<"entry">, MetaVarName<"<entry>">,
  HelpText<"Name of entry point symbol">;

defm error_limit:
  EEq<"error-limit", "Maximum number of errors to emit before stopping (0 = no limit)">;

def fatal_warnings: F<"fatal-warnings">,
  HelpText<"Treat warnings as errors">;

defm gc_sections: B<"gc-sections",
    "Enable garbage collection of unused sections",
    "Disable garbage collection of unused sections">;

defm merge_data_segments: BB<"merge-data-segments",
    "Enable merging data segments",
    "Disable merging data segments">;

def help: F<"help">, HelpText<"Print option help">;

def library: JoinedOrSeparate<["-"], "l">, MetaVarName<"<libName>">,
  HelpText<"Root name of library to use">;

def library_path: JoinedOrSeparate<["-"], "L">, MetaVarName<"<dir>">,
  HelpText<"Add a directory to the library search path">;

def m: JoinedOrSeparate<["-"], "m">, HelpText<"Set target emulation">;

defm mllvm: Eq<"mllvm", "Additional arguments to forward to LLVM's option processing">;

defm Map: Eq<"Map", "Print a link map to the specified file">;

def no_fatal_warnings: F<"no-fatal-warnings">;

def o: JoinedOrSeparate<["-"], "o">, MetaVarName<"<path>">,
  HelpText<"Path to file to write output">;

def O: JoinedOrSeparate<["-"], "O">, HelpText<"Optimize output file size">;

defm pie: B<"pie",
    "Create a position independent executable",
    "Do not create a position independent executable (default)">;

defm print_gc_sections: B<"print-gc-sections",
    "List removed unused sections",
    "Do not list removed unused sections">;

def print_map: F<"print-map">,
  HelpText<"Print a link map to the standard output">;

def relocatable: F<"relocatable">, HelpText<"Create relocatable object file">;

defm reproduce: EEq<"reproduce", "Dump linker invocation and input files for debugging">;

defm rsp_quoting: Eq<"rsp-quoting", "Quoting style for response files">,
  MetaVarName<"[posix,windows]">;

def shared: F<"shared">, HelpText<"Build a shared object">;

def strip_all: F<"strip-all">, HelpText<"Strip all symbols">;

def strip_debug: F<"strip-debug">, HelpText<"Strip debugging information">;

defm threads
    : Eq<"threads", "Number of threads. '1' disables multi-threading. By "
                    "default all available hardware threads are used">;

def trace: F<"trace">, HelpText<"Print the names of the input files">;

defm trace_symbol: Eq<"trace-symbol", "Trace references to symbols">;

defm undefined: Eq<"undefined", "Force undefined symbol during linking">;

defm unresolved_symbols:
  Eq<"unresolved-symbols", "Determine how to handle unresolved symbols">;

def v: Flag<["-"], "v">, HelpText<"Display the version number">;

def verbose: F<"verbose">, HelpText<"Verbose mode">;

def version: F<"version">, HelpText<"Display the version number and exit">;

def warn_unresolved_symbols: F<"warn-unresolved-symbols">,
  HelpText<"Report unresolved symbols as warnings">;

defm wrap: Eq<"wrap", "Use wrapper functions for symbol">,
  MetaVarName<"<symbol>=<symbol>">;

def z: JoinedOrSeparate<["-"], "z">, MetaVarName<"<option>">,
  HelpText<"Linker option extensions">;

// The follow flags are unique to wasm

def allow_undefined: F<"allow-undefined">,
  HelpText<"Allow undefined symbols in linked binary. "
           "This options is equivalent to --import-undefined "
           "and --unresolved-symbols=ignore-all">;

def import_undefined: F<"import-undefined">,
  HelpText<"Turn undefined symbols into imports where possible">;

def allow_undefined_file: J<"allow-undefined-file=">,
  HelpText<"Allow symbols listed in <file> to be undefined in linked binary">;

def allow_undefined_file_s: Separate<["-"], "allow-undefined-file">,
  Alias<allow_undefined_file>;

defm export: Eq<"export", "Force a symbol to be exported">;

defm export_if_defined: Eq<"export-if-defined",
     "Force a symbol to be exported, if it is defined in the input">;

def export_all: FF<"export-all">,
  HelpText<"Export all symbols (normally combined with --no-gc-sections)">;

def export_table: FF<"export-table">,
  HelpText<"Export function table to the environment">;

def growable_table: FF<"growable-table">,
  HelpText<"Remove maximum size from function table, allowing table to grow">;

def global_base: JJ<"global-base=">,
  HelpText<"Where to start to place global data">;

def import_memory: FF<"import-memory">,
  HelpText<"Import the module's memory from the default module of \"env\" with the name \"memory\".">;
def import_memory_with_name: JJ<"import-memory=">,
  HelpText<"Import the module's memory from the passed module with the passed name.">,
  MetaVarName<"<module>,<name>">;

def export_memory: FF<"export-memory">,
  HelpText<"Export the module's memory with the default name of \"memory\"">;
def export_memory_with_name: JJ<"export-memory=">,
  HelpText<"Export the module's memory with the passed name">;

def shared_memory: FF<"shared-memory">,
  HelpText<"Use shared linear memory">;

def import_table: FF<"import-table">,
  HelpText<"Import function table from the environment">;

def initial_memory: JJ<"initial-memory=">,
  HelpText<"Initial size of the linear memory">;

def max_memory: JJ<"max-memory=">,
  HelpText<"Maximum size of the linear memory">;

def no_entry: FF<"no-entry">,
  HelpText<"Do not output any entry point">;

def stack_first: FF<"stack-first">,
  HelpText<"Place stack at start of linear memory rather than after data">;

defm whole_archive: B<"whole-archive",
    "Force load of all members in a static library",
    "Do not force load of all members in a static library (default)">;

defm check_features: BB<"check-features",
    "Check feature compatibility of linked objects (default)",
    "Ignore feature compatibility of linked objects">;

def features: CommaJoined<["--", "-"], "features=">,
  HelpText<"Comma-separated used features, inferred from input objects by default.">;

def extra_features: CommaJoined<["--", "-"], "extra-features=">,
  HelpText<"Comma-separated list of features to add to the default set of features inferred from input objects.">;

// Aliases
def: JoinedOrSeparate<["-"], "e">, Alias<entry>;
def: J<"entry=">, Alias<entry>;
def: F<"call_shared">, Alias<Bdynamic>, HelpText<"Alias for --Bdynamic">;
def: F<"dy">, Alias<Bdynamic>, HelpText<"Alias for --Bdynamic">;
def: F<"dn">, Alias<Bstatic>, HelpText<"Alias for --Bstatic">;
def: F<"non_shared">, Alias<Bstatic>, HelpText<"Alias for --Bstatic">;
def: F<"static">, Alias<Bstatic>, HelpText<"Alias for --Bstatic">;
def: Flag<["-"], "E">, Alias<export_dynamic>, HelpText<"Alias for --export-dynamic">;
def: Flag<["-"], "i">, Alias<initial_memory>;
def: Separate<["--", "-"], "library">, Alias<library>;
def: Joined<["--", "-"], "library=">, Alias<library>;
def: Separate<["--", "-"], "library-path">, Alias<library_path>;
def: Joined<["--", "-"], "library-path=">, Alias<library_path>;
def: Flag<["-"], "M">, Alias<print_map>, HelpText<"Alias for --print-map">;
def: Flag<["-"], "r">, Alias<relocatable>;
def: Flag<["-"], "s">, Alias<strip_all>, HelpText<"Alias for --strip-all">;
def: Flag<["-"], "S">, Alias<strip_debug>, HelpText<"Alias for --strip-debug">;
def: Flag<["-"], "t">, Alias<trace>, HelpText<"Alias for --trace">;
def: JoinedOrSeparate<["-"], "y">, Alias<trace_symbol>, HelpText<"Alias for --trace-symbol">;
def: JoinedOrSeparate<["-"], "u">, Alias<undefined>;

// LTO-related options.
def lto_O: JJ<"lto-O">, MetaVarName<"<opt-level>">,
  HelpText<"Optimization level for LTO">;
def lto_partitions: JJ<"lto-partitions=">,
  HelpText<"Number of LTO codegen partitions">;
def disable_verify: F<"disable-verify">;
def save_temps: F<"save-temps">, HelpText<"Save intermediate LTO compilation results">;
def thinlto_cache_dir: JJ<"thinlto-cache-dir=">,
  HelpText<"Path to ThinLTO cached object file directory">;
defm thinlto_cache_policy: EEq<"thinlto-cache-policy", "Pruning policy for the ThinLTO cache">;
def thinlto_jobs: JJ<"thinlto-jobs=">,
  HelpText<"Number of ThinLTO jobs. Default to --threads=">;
def no_lto_legacy_pass_manager: FF<"no-lto-legacy-pass-manager">,
  HelpText<"Use new pass manager">;
def lto_debug_pass_manager: FF<"lto-debug-pass-manager">,
  HelpText<"Debug new pass manager">;

// Experimental PIC mode.
def experimental_pic: FF<"experimental-pic">,
  HelpText<"Enable Experimental PIC">;
