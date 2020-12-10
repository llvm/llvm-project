include "llvm/Option/OptParser.td"

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

multiclass BB<string name, string help1, string help2> {
  def NAME: Flag<["--"], name>, HelpText<help1>;
  def no_ # NAME: Flag<["--"], "no-" # name>, HelpText<help2>;
}

// The following flags are shared with the ELF linker
def Bsymbolic: F<"Bsymbolic">, HelpText<"Bind defined symbols locally">;

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

def error_limit: J<"error-limit=">,
  HelpText<"Maximum number of errors to emit before stopping (0 = no limit)">;

def fatal_warnings: F<"fatal-warnings">,
  HelpText<"Treat warnings as errors">;

defm gc_sections: B<"gc-sections",
    "Enable garbage collection of unused sections",
    "Disable garbage collection of unused sections">;

defm merge_data_segments: B<"merge-data-segments",
    "Enable merging data segments",
    "Disable merging data segments">;

def help: F<"help">, HelpText<"Print option help">;

def l: JoinedOrSeparate<["-"], "l">, MetaVarName<"<libName>">,
  HelpText<"Root name of library to use">;

def L: JoinedOrSeparate<["-"], "L">, MetaVarName<"<dir>">,
  HelpText<"Add a directory to the library search path">;

def m: JoinedOrSeparate<["-"], "m">, HelpText<"Set target emulation">;

def mllvm: S<"mllvm">, HelpText<"Options to pass to LLVM">;

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

defm reproduce: Eq<"reproduce", "Dump linker invocation and input files for debugging">;

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
  HelpText<"Allow undefined symbols in linked binary">;

def allow_undefined_file: J<"allow-undefined-file=">,
  HelpText<"Allow symbols listed in <file> to be undefined in linked binary">;

def allow_undefined_file_s: Separate<["-"], "allow-undefined-file">,
  Alias<allow_undefined_file>;

defm export: Eq<"export", "Force a symbol to be exported">;

def export_all: F<"export-all">,
  HelpText<"Export all symbols (normally combined with --no-gc-sections)">;

def export_table: F<"export-table">,
  HelpText<"Export function table to the environment">;

def growable_table: F<"growable-table">,
  HelpText<"Remove maximum size from function table, allowing table to grow">;

def global_base: J<"global-base=">,
  HelpText<"Where to start to place global data">;

def import_memory: F<"import-memory">,
  HelpText<"Import memory from the environment">;

def shared_memory: F<"shared-memory">,
  HelpText<"Use shared linear memory">;

def import_table: F<"import-table">,
  HelpText<"Import function table from the environment">;

def initial_memory: J<"initial-memory=">,
  HelpText<"Initial size of the linear memory">;

def max_memory: J<"max-memory=">,
  HelpText<"Maximum size of the linear memory">;

def no_entry: F<"no-entry">,
  HelpText<"Do not output any entry point">;

def stack_first: F<"stack-first">,
  HelpText<"Place stack at start of linear memory rather than after data">;

defm whole_archive: B<"whole-archive",
    "Force load of all members in a static library",
    "Do not force load of all members in a static library (default)">;

defm check_features: B<"check-features",
    "Check feature compatibility of linked objects (default)",
    "Ignore feature compatibility of linked objects">;

def features: CommaJoined<["--", "-"], "features=">,
  HelpText<"Comma-separated used features, inferred from input objects by default.">;

// Aliases
def: JoinedOrSeparate<["-"], "e">, Alias<entry>;
def: J<"entry=">, Alias<entry>;
def: Flag<["-"], "E">, Alias<export_dynamic>, HelpText<"Alias for --export-dynamic">;
def: Flag<["-"], "i">, Alias<initial_memory>;
def: Flag<["-"], "M">, Alias<print_map>, HelpText<"Alias for --print-map">;
def: Flag<["-"], "r">, Alias<relocatable>;
def: Flag<["-"], "s">, Alias<strip_all>, HelpText<"Alias for --strip-all">;
def: Flag<["-"], "S">, Alias<strip_debug>, HelpText<"Alias for --strip-debug">;
def: Flag<["-"], "t">, Alias<trace>, HelpText<"Alias for --trace">;
def: JoinedOrSeparate<["-"], "y">, Alias<trace_symbol>, HelpText<"Alias for --trace-symbol">;
def: JoinedOrSeparate<["-"], "u">, Alias<undefined>;

// LTO-related options.
def lto_O: J<"lto-O">, MetaVarName<"<opt-level>">,
  HelpText<"Optimization level for LTO">;
def lto_partitions: J<"lto-partitions=">,
  HelpText<"Number of LTO codegen partitions">;
def disable_verify: F<"disable-verify">;
def save_temps: F<"save-temps">;
def thinlto_cache_dir: J<"thinlto-cache-dir=">,
  HelpText<"Path to ThinLTO cached object file directory">;
defm thinlto_cache_policy: Eq<"thinlto-cache-policy", "Pruning policy for the ThinLTO cache">;
def thinlto_jobs: J<"thinlto-jobs=">,
  HelpText<"Number of ThinLTO jobs. Default to --threads=">;
defm lto_new_pass_manager: BB<"lto-new-pass-manager", "Use new pass manager", "Use legacy pass manager">;
def lto_debug_pass_manager: F<"lto-debug-pass-manager">,
  HelpText<"Debug new pass manager">;

// Experimental PIC mode.
def experimental_pic: F<"experimental-pic">,
  HelpText<"Enable Experimental PIC">;
