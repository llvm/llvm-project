include "llvm/Option/OptParser.td"

class F<string name>: Flag<["--", "-"], name>;
class J<string name>: Joined<["--", "-"], name>;
class S<string name>: Separate<["--", "-"], name>;

multiclass Eq<string name, string help> {
  def NAME: Separate<["--", "-"], name>;
  def NAME # _eq: Joined<["--", "-"], name # "=">, Alias<!cast<Separate>(NAME)>,
    HelpText<help>;
}

multiclass EqLong<string name, string help> {
  def NAME: Separate<["--"], name>;
  def NAME # _eq: Joined<["--"], name # "=">, Alias<!cast<Separate>(NAME)>,
    HelpText<help>;
}

multiclass B<string name, string help1, string help2> {
  def NAME: Flag<["--", "-"], name>, HelpText<help1>;
  def no_ # NAME: Flag<["--", "-"], "no-" # name>, HelpText<help2>;
}

def L: JoinedOrSeparate<["-"], "L">, MetaVarName<"<dir>">,
  HelpText<"Add a directory to the library search path">;
defm allow_multiple_definition: B<"allow-multiple-definition",
    "Allow multiple definitions",
    "Do not allow multiple definitions (default)">;
def Bdynamic: F<"Bdynamic">, HelpText<"Link against shared libraries">;
def Bstatic: F<"Bstatic">, HelpText<"Do not link against shared libraries">;
defm demangle: B<"demangle",
    "Demangle symbol names (default)",
    "Do not demangle symbol names">;
def disable_auto_import: F<"disable-auto-import">,
    HelpText<"Don't automatically import data symbols from other DLLs without dllimport">;
def disable_runtime_pseudo_reloc: F<"disable-runtime-pseudo-reloc">,
    HelpText<"Don't do automatic imports that require runtime fixups">;
def disable_stdcall_fixup: F<"disable-stdcall-fixup">,
    HelpText<"Don't resolve stdcall/fastcall/vectorcall to undecorated symbols">;
defm dynamicbase: B<"dynamicbase", "Enable ASLR", "Disable ASLR">;
def enable_auto_import: F<"enable-auto-import">,
    HelpText<"Automatically import data symbols from other DLLs where needed">;
def enable_runtime_pseudo_reloc: F<"enable-runtime-pseudo-reloc">,
    HelpText<"Allow automatic imports that require runtime fixups">;
def enable_stdcall_fixup: F<"enable-stdcall-fixup">,
    HelpText<"Resolve stdcall/fastcall/vectorcall to undecorated symbols without warnings">;
defm entry: Eq<"entry", "Name of entry point symbol">, MetaVarName<"<entry>">;
def exclude_all_symbols: F<"exclude-all-symbols">,
    HelpText<"Don't automatically export any symbols">;
def export_all_symbols: F<"export-all-symbols">,
    HelpText<"Export all symbols even if a def file or dllexport attributes are used">;
defm fatal_warnings: B<"fatal-warnings",
    "Treat warnings as errors",
    "Do not treat warnings as errors (default)">;
defm file_alignment: Eq<"file-alignment", "Set file alignment">;
defm gc_sections: B<"gc-sections",
    "Remove unused sections",
    "Don't remove unused sections">;
def help: F<"help">, HelpText<"Print option help">;
def icf: J<"icf=">, HelpText<"Identical code folding">;
def image_base: S<"image-base">, HelpText<"Base address of the program">;
defm insert_timestamp: B<"insert-timestamp",
    "Include PE header timestamp",
    "Don't include PE header timestamp">;
def kill_at: F<"kill-at">, HelpText<"Remove @n from exported symbols">;
def l: JoinedOrSeparate<["-"], "l">, MetaVarName<"<libName>">,
  HelpText<"Root name of library to use">;
def m: JoinedOrSeparate<["-"], "m">, HelpText<"Set target emulation">;
defm major_os_version: EqLong<"major-os-version",
     "Set the OS and subsystem major version">;
defm major_subsystem_version: EqLong<"major-subsystem-version",
     "Set the OS and subsystem major version">;
defm map: Eq<"Map", "Output a linker map">;
defm minor_os_version: EqLong<"minor-os-version",
     "Set the OS and subsystem minor version">;
defm minor_subsystem_version: EqLong<"minor-subsystem-version",
     "Set the OS and subsystem minor version">;
def no_seh: F<"no-seh">, HelpText<"Set the 'no SEH' flag in the executable">;
def large_address_aware: Flag<["--"], "large-address-aware">,
    HelpText<"Enable large addresses">;
def o: JoinedOrSeparate<["-"], "o">, MetaVarName<"<path>">,
  HelpText<"Path to file to write output">;
defm out_implib: Eq<"out-implib", "Import library name">;
defm output_def: Eq<"output-def", "Output def file">;
defm section_alignment: Eq<"section-alignment", "Set section alignment">;
def shared: F<"shared">, HelpText<"Build a shared object">;
defm subs: Eq<"subsystem", "Specify subsystem">;
defm stack: Eq<"stack", "Set size of the initial stack">;
def strip_all: F<"strip-all">,
    HelpText<"Omit all symbol information from the output binary">;
def strip_debug: F<"strip-debug">,
    HelpText<"Omit all debug information, but keep symbol information">;
defm reproduce: Eq<"reproduce",
     "Write a tar file containing input files and command line options to reproduce link">;
defm require_defined: Eq<"require-defined",
     "Force symbol to be added to symbol table as an undefined one">;
defm undefined: Eq<"undefined", "Include symbol in the link, if available">;
defm whole_archive: B<"whole-archive",
    "Include all object files for following archives",
    "No longer include all object files for following archives">;
def v: Flag<["-"], "v">, HelpText<"Display the version number">;
def verbose: F<"verbose">, HelpText<"Verbose mode">;
def version: F<"version">, HelpText<"Display the version number and exit">;
defm wrap: Eq<"wrap", "Use wrapper functions for symbol">,
     MetaVarName<"<symbol>">;

// LLD specific options
def _HASH_HASH_HASH : Flag<["-"], "###">,
    HelpText<"Print (but do not run) the commands to run for this compilation">;
def appcontainer: F<"appcontainer">, HelpText<"Set the appcontainer flag in the executable">;
defm delayload: Eq<"delayload", "DLL to load only on demand">;
def mllvm: S<"mllvm">;
defm pdb: Eq<"pdb", "Output PDB debug info file, chosen implicitly if the argument is empty">;
defm thinlto_cache_dir: EqLong<"thinlto-cache-dir",
  "Path to ThinLTO cached object file directory">;
def Xlink : J<"Xlink=">, MetaVarName<"<arg>">,
    HelpText<"Pass <arg> to the COFF linker">;

// Alias
def alias_Bdynamic_call_shared: Flag<["-"], "call_shared">, Alias<Bdynamic>;
def alias_Bdynamic_dy: Flag<["-"], "dy">, Alias<Bdynamic>;
def alias_Bstatic_dn: Flag<["-"], "dn">, Alias<Bstatic>;
def alias_Bstatic_non_shared: Flag<["-"], "non_shared">, Alias<Bstatic>;
def alias_Bstatic_static: Flag<["-"], "static">, Alias<Bstatic>;
def alias_entry_e: JoinedOrSeparate<["-"], "e">, Alias<entry>;
def alias_strip_s: Flag<["-"], "s">, Alias<strip_all>;
def alias_strip_S: Flag<["-"], "S">, Alias<strip_debug>;
def alias_undefined_u: JoinedOrSeparate<["-"], "u">, Alias<undefined>;

// Ignored options
def: Joined<["-"], "O">;
def: F<"build-id">;
def: F<"disable-auto-image-base">;
def: F<"enable-auto-image-base">;
def: F<"end-group">;
def: Flag<["--"], "full-shutdown">;
def: F<"high-entropy-va">;
def: S<"major-image-version">;
def: S<"minor-image-version">;
def: F<"no-undefined">;
def: F<"nxcompat">;
def: F<"pic-executable">;
def: S<"plugin">;
def: J<"plugin=">;
def: S<"plugin-opt">;
def: J<"plugin-opt=">;
def: J<"sysroot">;
def: F<"start-group">;
def: F<"tsaware">;
