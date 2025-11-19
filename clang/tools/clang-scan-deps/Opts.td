include "llvm/Option/OptParser.td"

class F<string name, string help> : Flag<["-"], name>, HelpText<help>;
class Arg<string name, string help> : Separate<["-"], name>, HelpText<help>;

multiclass Eq<string name, string help> {
    def NAME #_EQ : Joined<["-", "--"], name #"=">, HelpText<help>;
    def : Separate<["-", "--"], name>, Alias<!cast<Joined>(NAME #_EQ)>;
}

def help : Flag<["--"], "help">, HelpText<"Display this help">;
def version : Flag<["--"], "version">, HelpText<"Display the version">;

def o : Arg<"o", "Destination of the primary output">;

defm mode : Eq<"mode", "The preprocessing mode used to compute the dependencies">;

defm format : Eq<"format", "The output format for the dependencies">;

defm module_files_dir : Eq<"module-files-dir",
    "The build directory for modules. Defaults to the value of '-fmodules-cache-path=' from command lines for implicit modules">;

def optimize_args_EQ : CommaJoined<["-", "--"], "optimize-args=">, HelpText<"Which command-line arguments of modules to optimize">;
def eager_load_pcm : F<"eager-load-pcm", "Load PCM files eagerly (instead of lazily on import)">;
def cache_negative_stats : F<"cache-negative-stats", "Cache stat failures">;

def j : Arg<"j", "Number of worker threads to use (default: use all concurrent threads)">;

defm compilation_database : Eq<"compilation-database", "Compilation database">;
defm module_names
    : Eq<"module-names", "A comma separated list of names of modules of which "
                         "the dependencies are to be computed">;
defm dependency_target : Eq<"dependency-target", "The names of dependency targets for the dependency file">;

defm tu_buffer_path: Eq<"tu-buffer-path", "The path to the translation unit for depscan. Not compatible with -module-name">;

def deprecated_driver_command : F<"deprecated-driver-command", "use a single driver command to build the tu (deprecated)">;

defm resource_dir_recipe : Eq<"resource-dir-recipe", "How to produce missing '-resource-dir' argument">;

def emit_cas_compdb : F<"emit-cas-compdb", "Emit compilation DB with updated clang arguments for CAS based dependency scanning build.">;

defm cas_path : Eq<"cas-path", "Path for on-disk CAS.">;

defm fcas_plugin_path : Eq<"fcas-plugin-path", "Path to a shared library implementing the LLVM CAS plugin API">;
defm fcas_plugin_option : Eq<"fcas-plugin-option", "Option to pass to the CAS plugin">;

def in_memory_cas : F<"in-memory-cas", "Use an in-memory CAS instead of on-disk.">;

defm prefix_map_toolchain : Eq<"prefix-map-toolchain", "Path to remap toolchain path to.">;

defm prefix_map_sdk : Eq<"prefix-map-sdk", "Path to remap SDK path to.">;

defm prefix_map : Eq<"prefix-map", "Path to remap, as \"<old>=<new>\".">;

def print_timing : F<"print-timing", "Print timing information">;

def emit_visible_modules
    : F<"emit-visible-modules", "emit visible modules in primary output">;

def verbose : F<"v", "Use verbose output">;

def round_trip_args : F<"round-trip-args", "verify that command-line arguments are canonical by parsing and re-serializing">;

def verbatim_args : F<"verbatim-args", "Pass commands to the scanner verbatim without adjustments">;

def DASH_DASH : Option<["--"], "", KIND_REMAINING_ARGS>;
