include "llvm/Option/OptParser.td"

// For options whose names are multiple letters, either one dash or
// two can precede the option name except those that start with 'o'.
class F<string name>: Flag<["--", "-"], name>;
class J<string name>: Joined<["--", "-"], name>;

multiclass Eq<string name> {
  def NAME: Separate<["--", "-"], name>;
  def NAME # _eq: Joined<["--", "-"], name # "=">, Alias<!cast<Separate>(NAME)>;
}

multiclass B<string name, string help1, string help2> {
  def NAME: Flag<["--", "-"], name>, HelpText<help1>;
  def no_ # NAME: Flag<["--", "-"], "no-" # name>, HelpText<help2>;
}

defm auxiliary: Eq<"auxiliary">,
  HelpText<"Set DT_AUXILIARY field to the specified name">;

def Bsymbolic: F<"Bsymbolic">, HelpText<"Bind defined symbols locally">;

def Bsymbolic_functions: F<"Bsymbolic-functions">,
  HelpText<"Bind defined function symbols locally">;

def Bdynamic: F<"Bdynamic">, HelpText<"Link against shared libraries">;

def Bstatic: F<"Bstatic">, HelpText<"Do not link against shared libraries">;

def build_id: F<"build-id">, HelpText<"Generate build ID note">;

def build_id_eq: J<"build-id=">, HelpText<"Generate build ID note">;

defm check_sections : B<"check-sections",
    "Check section addresses for overlaps",
    "Do not check section addresses for overlaps">;

defm compress_debug_sections : Eq<"compress-debug-sections">,
  HelpText<"Compress DWARF debug sections">;

defm defsym: Eq<"defsym">, HelpText<"Define a symbol alias">;

defm library_path: Eq<"library-path">,
  HelpText<"Add a directory to the library search path">, MetaVarName<"<dir>">;

def O: JoinedOrSeparate<["-"], "O">, HelpText<"Optimize output file size">;

defm Tbss: Eq<"Tbss">,
  HelpText<"Same as --section-start with .bss as the sectionname">;

defm Tdata: Eq<"Tdata">,
  HelpText<"Same as --section-start with .data as the sectionname">;

defm Ttext: Eq<"Ttext">,
  HelpText<"Same as --section-start with .text as the sectionname">;

defm allow_multiple_definition: B<"allow-multiple-definition",
    "Allow multiple definitions",
    "Do not allow multiple definitions">;

defm apply_dynamic_relocs: B<"apply-dynamic-relocs",
    "Apply dynamic relocations to place",
    "Do not apply dynamic relocations to place">;

defm as_needed: B<"as-needed",
    "Only set DT_NEEDED for shared libraries if used",
    "Always set DT_NEEDED for shared libraries">;

defm call_graph_ordering_file: Eq<"call-graph-ordering-file">,
  HelpText<"Layout sections to optimize the given callgraph">;

// -chroot doesn't have a help text because it is an internal option.
defm chroot: Eq<"chroot">;

def color_diagnostics: F<"color-diagnostics">,
  HelpText<"Use colors in diagnostics">;

def color_diagnostics_eq: J<"color-diagnostics=">,
  HelpText<"Use colors in diagnostics">;

defm cref: B<"cref",
    "Output cross reference table",
    "Do not output cross reference table">;

defm define_common: B<"define-common",
    "Assign space to common symbols",
    "Do not assign space to common symbols">;

defm demangle: B<"demangle",
    "Demangle symbol names",
    "Do not demangle symbol names">;

def disable_new_dtags: F<"disable-new-dtags">,
  HelpText<"Disable new dynamic tags">;

def discard_all: F<"discard-all">, HelpText<"Delete all local symbols">;

def discard_locals: F<"discard-locals">,
  HelpText<"Delete temporary local symbols">;

def discard_none: F<"discard-none">,
  HelpText<"Keep all symbols in the symbol table">;

defm dynamic_linker: Eq<"dynamic-linker">,
  HelpText<"Which dynamic linker to use">;

defm dynamic_list: Eq<"dynamic-list">,
  HelpText<"Read a list of dynamic symbols">;

defm eh_frame_hdr: B<"eh-frame-hdr",
    "Request creation of .eh_frame_hdr section and PT_GNU_EH_FRAME segment header",
    "Do not create .eh_frame_hdr section">;

def emit_relocs: F<"emit-relocs">, HelpText<"Generate relocations in output">;

def enable_new_dtags: F<"enable-new-dtags">,
  HelpText<"Enable new dynamic tags">;

def end_group: F<"end-group">,
  HelpText<"Ignored for compatibility with GNU unless you pass --warn-backrefs">;

def end_lib: F<"end-lib">,
  HelpText<"End a grouping of objects that should be treated as if they were together in an archive">;

defm entry: Eq<"entry">, HelpText<"Name of entry point symbol">,
  MetaVarName<"<entry>">;

defm error_limit: Eq<"error-limit">,
  HelpText<"Maximum number of errors to emit before stopping (0 = no limit)">;

def error_unresolved_symbols: F<"error-unresolved-symbols">,
  HelpText<"Report unresolved symbols as errors">;

defm exclude_libs: Eq<"exclude-libs">,
  HelpText<"Exclude static libraries from automatic export">;

defm export_dynamic: B<"export-dynamic",
    "Put symbols in the dynamic symbol table",
    "Do not put symbols in the dynamic symbol table">;

defm export_dynamic_symbol: Eq<"export-dynamic-symbol">,
  HelpText<"Put a symbol in the dynamic symbol table">;

defm fatal_warnings: B<"fatal-warnings",
    "Treat warnings as errors",
    "Do not treat warnings as errors">;

defm filter: Eq<"filter">,
  HelpText<"Set DT_FILTER field to the specified name">;

defm fini: Eq<"fini">,
  HelpText<"Specify a finalizer function">, MetaVarName<"<symbol>">;

def fix_cortex_a53_843419: F<"fix-cortex-a53-843419">,
  HelpText<"Apply fixes for AArch64 Cortex-A53 erratum 843419">;

defm format: Eq<"format">,
  HelpText<"Change the input format of the inputs following this option">,
  MetaVarName<"<input-format>">;

defm gc_sections: B<"gc-sections",
    "Enable garbage collection of unused sections",
    "Disable garbage collection of unused sections">;

defm gdb_index: B<"gdb-index",
    "Generate .gdb_index section",
    "Do not generate .gdb_index section">;

defm gnu_unique: B<"gnu-unique",
  "Enable STB_GNU_UNIQUE symbol binding",
  "Disable STB_GNU_UNIQUE symbol binding">;

defm hash_style: Eq<"hash-style">,
  HelpText<"Specify hash style (sysv, gnu or both)">;

def help: F<"help">, HelpText<"Print option help">;

def icf_all: F<"icf=all">, HelpText<"Enable identical code folding">;

def icf_none: F<"icf=none">, HelpText<"Disable identical code folding">;

def ignore_function_address_equality: F<"ignore-function-address-equality">,
  HelpText<"lld can break the address equality of functions">;

def ignore_data_address_equality: F<"ignore-data-address-equality">,
  HelpText<"lld can break the address equality of data">;

defm image_base : Eq<"image-base">, HelpText<"Set the base address">;

defm init: Eq<"init">, HelpText<"Specify an initializer function">,
  MetaVarName<"<symbol>">;

defm just_symbols: Eq<"just-symbols">, HelpText<"Just link symbols">;

defm library: Eq<"library">, HelpText<"Root name of library to use">,
  MetaVarName<"<libName>">;

def lto_O: J<"lto-O">, MetaVarName<"<opt-level>">,
  HelpText<"Optimization level for LTO">;

def m: JoinedOrSeparate<["-"], "m">, HelpText<"Set target emulation">;

defm Map: Eq<"Map">, HelpText<"Print a link map to the specified file">;

defm merge_exidx_entries: B<"merge-exidx-entries",
    "Enable merging .ARM.exidx entries",
    "Disable merging .ARM.exidx entries">;

def nostdlib: F<"nostdlib">,
  HelpText<"Only search directories specified on the command line">;

def no_color_diagnostics: F<"no-color-diagnostics">,
  HelpText<"Do not use colors in diagnostics">;

def no_dynamic_linker: F<"no-dynamic-linker">,
  HelpText<"Inhibit output of .interp section">;

def noinhibit_exec: F<"noinhibit-exec">,
  HelpText<"Retain the executable output file whenever it is still usable">;

def no_omagic: F<"no-omagic">, MetaVarName<"<magic>">,
  HelpText<"Do not set the text data sections to be writable">;

def no_rosegment: F<"no-rosegment">,
  HelpText<"Do not put read-only non-executable sections in their own segment">;

def no_undefined: F<"no-undefined">,
  HelpText<"Report unresolved symbols even if the linker is creating a shared library">;

def o: JoinedOrSeparate<["-"], "o">, MetaVarName<"<path>">,
  HelpText<"Path to file to write output">;

def oformat: Separate<["--"], "oformat">, MetaVarName<"<format>">,
  HelpText<"Specify the binary format for the output object file">;

def omagic: Flag<["--"], "omagic">, MetaVarName<"<magic>">,
  HelpText<"Set the text and data sections to be readable and writable">;

defm orphan_handling: Eq<"orphan-handling">,
  HelpText<"Control how orphan sections are handled when linker script used">;

defm pack_dyn_relocs: Eq<"pack-dyn-relocs">, MetaVarName<"<format>">,
  HelpText<"Pack dynamic relocations in the given format (none or android)">;

defm pie: B<"pie",
    "Create a position independent executable",
    "Do not create a position independent executable">;

defm print_gc_sections: B<"print-gc-sections",
    "List removed unused sections",
    "Do not list removed unused sections">;

defm print_icf_sections: B<"print-icf-sections",
    "List identical folded sections",
    "Do not list identical folded sections">;

def print_map: F<"print-map">,
  HelpText<"Print a link map to the standard output">;

defm reproduce: Eq<"reproduce">,
  HelpText<"Dump linker invocation and input files for debugging">;

defm rpath: Eq<"rpath">, HelpText<"Add a DT_RUNPATH to the output">;

def relocatable: F<"relocatable">, HelpText<"Create relocatable object file">;

defm retain_symbols_file: Eq<"retain-symbols-file">,
  HelpText<"Retain only the symbols listed in the file">,
  MetaVarName<"<file>">;

defm script: Eq<"script">, HelpText<"Read linker script">;

defm section_start: Eq<"section-start">, MetaVarName<"<address>">,
  HelpText<"Set address of section">;

def shared: F<"shared">, HelpText<"Build a shared object">;

defm soname: Eq<"soname">, HelpText<"Set DT_SONAME">;

defm sort_section: Eq<"sort-section">,
  HelpText<"Specifies sections sorting rule when linkerscript is used">;

def start_group: F<"start-group">,
  HelpText<"Ignored for compatibility with GNU unless you pass --warn-backrefs">;

def start_lib: F<"start-lib">,
  HelpText<"Start a grouping of objects that should be treated as if they were together in an archive">;

def strip_all: F<"strip-all">, HelpText<"Strip all symbols">;

def strip_debug: F<"strip-debug">, HelpText<"Strip debugging information">;

defm symbol_ordering_file: Eq<"symbol-ordering-file">,
  HelpText<"Layout sections to place symbols in the order specified by symbol ordering file">;

defm sysroot: Eq<"sysroot">, HelpText<"Set the system root">;

def target1_rel: F<"target1-rel">, HelpText<"Interpret R_ARM_TARGET1 as R_ARM_REL32">;

def target1_abs: F<"target1-abs">, HelpText<"Interpret R_ARM_TARGET1 as R_ARM_ABS32">;

defm target2: Eq<"target2">,
  HelpText<"Interpret R_ARM_TARGET2 as <type>, where <type> is one of rel, abs, or got-rel">,
  MetaVarName<"<type>">;

defm threads: B<"threads",
    "Run the linker multi-threaded",
    "Do not run the linker multi-threaded">;

def trace: F<"trace">, HelpText<"Print the names of the input files">;

defm trace_symbol : Eq<"trace-symbol">, HelpText<"Trace references to symbols">;

defm undefined: Eq<"undefined">,
  HelpText<"Force undefined symbol during linking">;

defm unresolved_symbols: Eq<"unresolved-symbols">,
  HelpText<"Determine how to handle unresolved symbols">;

defm undefined_version: B<"undefined-version",
  "Allow unused version in version script",
  "Report version scripts that refer undefined symbols">;

defm rsp_quoting: Eq<"rsp-quoting">,
  HelpText<"Quoting style for response files. Values supported: windows|posix">;

def v: Flag<["-"], "v">, HelpText<"Display the version number">;

def verbose: F<"verbose">, HelpText<"Verbose mode">;

def version: F<"version">, HelpText<"Display the version number and exit">;

defm version_script: Eq<"version-script">, HelpText<"Read a version script">;

defm warn_backrefs: B<"warn-backrefs",
    "Warn about backward symbol references to fetch archive members",
    "Do not warn about backward symbol references to fetch archive members">;

defm warn_common: B<"warn-common",
    "Warn about duplicate common symbols",
    "Do not warn about duplicate common symbols">;

defm warn_symbol_ordering : B<"warn-symbol-ordering",
    "Warn about problems with the symbol ordering file",
    "Do not warn about problems with the symbol ordering file">;

def warn_unresolved_symbols: F<"warn-unresolved-symbols">,
  HelpText<"Report unresolved symbols as warnings">;

defm whole_archive: B<"whole-archive",
    "Force load of all members in a static library",
    "Do not force load of all members in a static library">;

defm wrap: Eq<"wrap">, HelpText<"Use wrapper functions for symbol">,
  MetaVarName<"<symbol>">;

def z: JoinedOrSeparate<["-"], "z">, MetaVarName<"<option>">,
  HelpText<"Linker option extensions">;

// Aliases
def alias_auxiliary: Separate<["-"], "f">, Alias<auxiliary>;
def alias_Bdynamic_call_shared: F<"call_shared">, Alias<Bdynamic>;
def alias_Bdynamic_dy: F<"dy">, Alias<Bdynamic>;
def alias_Bstatic_dn: F<"dn">, Alias<Bstatic>;
def alias_Bstatic_non_shared: F<"non_shared">, Alias<Bstatic>;
def alias_Bstatic_static: F<"static">, Alias<Bstatic>;
def alias_define_common_d: Flag<["-"], "d">, Alias<define_common>;
def alias_define_common_dc: F<"dc">, Alias<define_common>;
def alias_define_common_dp: F<"dp">, Alias<define_common>;
def alias_discard_all_x: Flag<["-"], "x">, Alias<discard_all>;
def alias_discard_locals_X: Flag<["-"], "X">, Alias<discard_locals>;
def alias_emit_relocs: Flag<["-"], "q">, Alias<emit_relocs>;
def alias_end_group_paren: Flag<["-"], ")">, Alias<end_group>;
def alias_entry_e: JoinedOrSeparate<["-"], "e">, Alias<entry>;
def alias_export_dynamic_E: Flag<["-"], "E">, Alias<export_dynamic>;
def alias_filter: Separate<["-"], "F">, Alias<filter>;
def alias_format_b: Separate<["-"], "b">, Alias<format>;
def alias_library: JoinedOrSeparate<["-"], "l">, Alias<library>;
def alias_library_path: JoinedOrSeparate<["-"], "L">, Alias<library_path>;
def alias_no_pie_pic_executable: F<"no-pic-executable">, Alias<no_pie>;
def alias_omagic: Flag<["-"], "N">, Alias<omagic>;
def alias_o_output: Joined<["--"], "output=">, Alias<o>;
def alias_o_output2 : Separate<["--"], "output">, Alias<o>;
def alias_pie_pic_executable: F<"pic-executable">, Alias<pie>;
def alias_print_map_M: Flag<["-"], "M">, Alias<print_map>;
def alias_relocatable_r: Flag<["-"], "r">, Alias<relocatable>;
def alias_rpath_R: JoinedOrSeparate<["-"], "R">, Alias<rpath>;
def alias_script_T: JoinedOrSeparate<["-"], "T">, Alias<script>;
def alias_shared_Bshareable: F<"Bshareable">, Alias<shared>;
def alias_soname_h: JoinedOrSeparate<["-"], "h">, Alias<soname>;
def alias_start_group_paren: Flag<["-"], "(">, Alias<start_group>;
def alias_strip_all: Flag<["-"], "s">, Alias<strip_all>;
def alias_strip_debug_S: Flag<["-"], "S">, Alias<strip_debug>;
def alias_trace: Flag<["-"], "t">, Alias<trace>;
def alias_trace_symbol_y : JoinedOrSeparate<["-"], "y">, Alias<trace_symbol>;
def alias_Ttext_segment: Separate<["-", "--"], "Ttext-segment">, Alias<Ttext>;
def alias_Ttext_segment_eq: Joined<["-", "--"], "Ttext-segment=">, Alias<Ttext>;
def alias_undefined_u: JoinedOrSeparate<["-"], "u">, Alias<undefined>;
def alias_version_V: Flag<["-"], "V">, Alias<version>;

// LTO-related options.
def lto_aa_pipeline: J<"lto-aa-pipeline=">,
  HelpText<"AA pipeline to run during LTO. Used in conjunction with -lto-newpm-passes">;
def lto_debug_pass_manager: F<"lto-debug-pass-manager">,
  HelpText<"Debug new pass manager">;
def lto_new_pass_manager: F<"lto-new-pass-manager">,
  HelpText<"Use new pass manager">;
def lto_newpm_passes: J<"lto-newpm-passes=">,
  HelpText<"Passes to run during LTO">;
def lto_partitions: J<"lto-partitions=">,
  HelpText<"Number of LTO codegen partitions">;
def lto_sample_profile: J<"lto-sample-profile=">,
  HelpText<"Sample profile file path">;
def disable_verify: F<"disable-verify">;
defm mllvm: Eq<"mllvm">;
def opt_remarks_filename: Separate<["--"], "opt-remarks-filename">,
  HelpText<"YAML output file for optimization remarks">;
def opt_remarks_with_hotness: Flag<["--"], "opt-remarks-with-hotness">,
  HelpText<"Include hotness information in the optimization remarks file">;
defm plugin_opt: Eq<"plugin-opt">,
  HelpText<"specifies LTO options for compatibility with GNU linkers">;
def save_temps: F<"save-temps">;
def thinlto_cache_dir: J<"thinlto-cache-dir=">,
  HelpText<"Path to ThinLTO cached object file directory">;
defm thinlto_cache_policy: Eq<"thinlto-cache-policy">,
  HelpText<"Pruning policy for the ThinLTO cache">;
def thinlto_jobs: J<"thinlto-jobs=">, HelpText<"Number of ThinLTO jobs">;

// Ignore LTO plugin-related options.
// clang -flto passes -plugin and -plugin-opt to the linker. This is required
// for ld.gold and ld.bfd to get LTO working. But it's not for lld which doesn't
// rely on a plugin. Instead of detecting which linker is used on clang side we
// just ignore the option on lld side as it's easier. In fact, the linker could
// be called 'ld' and understanding which linker is used would require parsing of
// --version output.
defm plugin: Eq<"plugin">;

// Options listed below are silently ignored for now for compatibility.
def allow_shlib_undefined: F<"allow-shlib-undefined">;
def detect_odr_violations: F<"detect-odr-violations">;
def g: Flag<["-"], "g">;
def long_plt: F<"long-plt">;
def no_add_needed: F<"no-add-needed">;
def no_allow_shlib_undefined: F<"no-allow-shlib-undefined">;
def no_copy_dt_needed_entries: F<"no-copy-dt-needed-entries">;
def no_ctors_in_init_array: F<"no-ctors-in-init-array">;
def no_keep_memory: F<"no-keep-memory">;
def no_mmap_output_file: F<"no-mmap-output-file">;
def no_warn_mismatch: F<"no-warn-mismatch">;
defm rpath_link: Eq<"rpath-link">;
def sort_common: F<"sort-common">;
def stats: F<"stats">;
def warn_execstack: F<"warn-execstack">;
def warn_once: F<"warn-once">;
def warn_shared_textrel: F<"warn-shared-textrel">;
def EB : F<"EB">;
def EL : F<"EL">;
def G: JoinedOrSeparate<["-"], "G">;
def Qy : F<"Qy">;
