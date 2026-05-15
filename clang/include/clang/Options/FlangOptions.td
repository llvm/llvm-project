//===--- FlangOptions.td - Flang-specific options ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def flang_ignored_w_Group : OptionGroup<"<flang ignored W group>">,
  Group<W_Group>, Flags<[Ignored]>, Visibility<[FlangOption]>;

multiclass FlangIgnoredDiagOpt<string name> {
  def unsupported_warning_w#NAME : Flag<["-", "--"], "W"#name>,
    Visibility<[FlangOption]>, Group<flang_ignored_w_Group>;
}

// Generic gfortran options.
def A_DASH : Joined<["-"], "A-">, Group<gfortran_Group>;
def static_libgfortran : Flag<["-"], "static-libgfortran">, Group<gfortran_Group>;

// "f" options with values for gfortran.
def fblas_matmul_limit_EQ : Joined<["-"], "fblas-matmul-limit=">, Group<gfortran_Group>;
def fcheck_EQ : Joined<["-"], "fcheck=">, Group<gfortran_Group>;
def ffpe_trap_EQ : Joined<["-"], "ffpe-trap=">, Group<gfortran_Group>;
def ffree_line_length_VALUE : Joined<["-"], "ffree-line-length-">, Group<gfortran_Group>;
def finit_character_EQ : Joined<["-"], "finit-character=">, Group<gfortran_Group>;
def finit_integer_EQ : Joined<["-"], "finit-integer=">, Group<gfortran_Group>;
def finit_logical_EQ : Joined<["-"], "finit-logical=">, Group<gfortran_Group>;
def finit_real_EQ : Joined<["-"], "finit-real=">, Group<gfortran_Group>;
def fmax_array_constructor_EQ : Joined<["-"], "fmax-array-constructor=">, Group<gfortran_Group>;
def fmax_errors_EQ : Joined<["-"], "fmax-errors=">, Group<gfortran_Group>;
def fmax_stack_var_size_EQ : Joined<["-"], "fmax-stack-var-size=">, Group<gfortran_Group>;
def fmax_subrecord_length_EQ : Joined<["-"], "fmax-subrecord-length=">, Group<gfortran_Group>;
def frecord_marker_EQ : Joined<["-"], "frecord-marker=">, Group<gfortran_Group>;

// "f" flags for gfortran.
defm aggressive_function_elimination : BooleanFFlag<"aggressive-function-elimination">, Group<gfortran_Group>;
defm align_commons : BooleanFFlag<"align-commons">, Group<gfortran_Group>;
defm all_intrinsics : BooleanFFlag<"all-intrinsics">, Group<gfortran_Group>;
def fautomatic : Flag<["-"], "fautomatic">; // -fno-automatic is significant
defm backtrace : BooleanFFlag<"backtrace">, Group<gfortran_Group>;
defm bounds_check : BooleanFFlag<"bounds-check">, Group<gfortran_Group>;
defm check_array_temporaries : BooleanFFlag<"check-array-temporaries">, Group<gfortran_Group>;
defm cray_pointer : BooleanFFlag<"cray-pointer">, Group<gfortran_Group>;
defm d_lines_as_code : BooleanFFlag<"d-lines-as-code">,
                       HelpText<"Treat fixed form lines with 'd' or 'D' in the "
                                "first column as blank.">,
                       Group<f_Group>,
                       Visibility<[FlangOption, FC1Option]>;
defm d_lines_as_comments : BooleanFFlag<"d-lines-as-comments">,
                           HelpText<"Treat fixed form lines with 'd' or 'D' in "
                                    "the first column as comments.">,
                           Group<f_Group>,
                           Visibility<[FlangOption, FC1Option]>;
defm dollar_ok : BooleanFFlag<"dollar-ok">, Group<gfortran_Group>;
defm dump_fortran_optimized : BooleanFFlag<"dump-fortran-optimized">, Group<gfortran_Group>;
defm dump_fortran_original : BooleanFFlag<"dump-fortran-original">, Group<gfortran_Group>;
defm dump_parse_tree : BooleanFFlag<"dump-parse-tree">, Group<gfortran_Group>;
defm external_blas : BooleanFFlag<"external-blas">, Group<gfortran_Group>;
defm f2c : BooleanFFlag<"f2c">, Group<gfortran_Group>;
defm frontend_optimize : BooleanFFlag<"frontend-optimize">, Group<gfortran_Group>;
defm init_local_zero : BooleanFFlag<"init-local-zero">, Group<gfortran_Group>;
defm integer_4_integer_8 : BooleanFFlag<"integer-4-integer-8">, Group<gfortran_Group>;
defm max_identifier_length : BooleanFFlag<"max-identifier-length">, Group<gfortran_Group>;
defm module_private : BooleanFFlag<"module-private">, Group<gfortran_Group>;
defm pack_derived : BooleanFFlag<"pack-derived">, Group<gfortran_Group>;
//defm protect_parens : BooleanFFlag<"protect-parens">, Group<gfortran_Group>;
defm range_check : BooleanFFlag<"range-check">, Group<gfortran_Group>;
defm real_4_real_10 : BooleanFFlag<"real-4-real-10">, Group<gfortran_Group>;
defm real_4_real_16 : BooleanFFlag<"real-4-real-16">, Group<gfortran_Group>;
defm real_4_real_8 : BooleanFFlag<"real-4-real-8">, Group<gfortran_Group>;
defm real_8_real_10 : BooleanFFlag<"real-8-real-10">, Group<gfortran_Group>;
defm real_8_real_16 : BooleanFFlag<"real-8-real-16">, Group<gfortran_Group>;
defm real_8_real_4 : BooleanFFlag<"real-8-real-4">, Group<gfortran_Group>;
defm recursive : BooleanFFlag<"recursive">, Group<gfortran_Group>;
defm second_underscore : BooleanFFlag<"second-underscore">, Group<gfortran_Group>;
defm sign_zero : BooleanFFlag<"sign-zero">, Group<gfortran_Group>;
defm whole_file : BooleanFFlag<"whole-file">, Group<gfortran_Group>;

// -W <arg> options unsupported by flang.
// If any of these options are passed to flang's driver,
// a warning will be raised and the argument will be claimed.
defm : FlangIgnoredDiagOpt<"extra">;
defm : FlangIgnoredDiagOpt<"aliasing">;
defm : FlangIgnoredDiagOpt<"ampersand">;
defm : FlangIgnoredDiagOpt<"array-bounds">;
defm : FlangIgnoredDiagOpt<"c-binding-type">;
defm : FlangIgnoredDiagOpt<"character-truncation">;
defm : FlangIgnoredDiagOpt<"conversion">;
defm : FlangIgnoredDiagOpt<"do-subscript">;
defm : FlangIgnoredDiagOpt<"function-elimination">;
defm : FlangIgnoredDiagOpt<"implicit-interface">;
defm : FlangIgnoredDiagOpt<"implicit-procedure">;
defm : FlangIgnoredDiagOpt<"intrinsic-shadow">;
defm : FlangIgnoredDiagOpt<"use-without-only">;
defm : FlangIgnoredDiagOpt<"intrinsics-std">;
defm : FlangIgnoredDiagOpt<"line-truncation">;
defm : FlangIgnoredDiagOpt<"no-align-commons">;
defm : FlangIgnoredDiagOpt<"no-overwrite-recursive">;
defm : FlangIgnoredDiagOpt<"no-tabs">;
defm : FlangIgnoredDiagOpt<"real-q-constant">;
defm : FlangIgnoredDiagOpt<"surprising">;
defm : FlangIgnoredDiagOpt<"underflow">;
defm : FlangIgnoredDiagOpt<"unused-parameter">;
defm : FlangIgnoredDiagOpt<"realloc-lhs">;
defm : FlangIgnoredDiagOpt<"realloc-lhs-all">;
defm : FlangIgnoredDiagOpt<"frontend-loop-interchange">;
defm : FlangIgnoredDiagOpt<"target-lifetime">;

def shared_libflangrt : Flag<["-"], "shared-libflangrt">,
  HelpText<"Link the flang-rt shared library">, Group<Link_Group>,
  Visibility<[FlangOption]>, Flags<[NoArgumentUnused]>;
def static_libflangrt : Flag<["-"], "static-libflangrt">,
  HelpText<"Link the flang-rt static library">, Group<Link_Group>,
  Visibility<[FlangOption]>, Flags<[NoArgumentUnused]>;

//===----------------------------------------------------------------------===//
// FlangOption + CoreOption + NoXarchOption
//===----------------------------------------------------------------------===//

def Xflang : Separate<["-"], "Xflang">,
  HelpText<"Pass <arg> to the flang compiler">, MetaVarName<"<arg>">,
  Flags<[NoXarchOption]>, Visibility<[FlangOption]>,
  Group<CompileOnly_Group>;

//===----------------------------------------------------------------------===//
// FlangOption and FC1 Options
//===----------------------------------------------------------------------===//

let Visibility = [FC1Option, FlangOption] in {

def cpp : Flag<["-"], "cpp">, Group<f_Group>,
  HelpText<"Enable predefined and command line preprocessor macros">;
def nocpp : Flag<["-"], "nocpp">, Group<f_Group>,
  HelpText<"Disable predefined and command line preprocessor macros">;
def module_dir : JoinedOrSeparate<["-"], "module-dir">, MetaVarName<"<dir>">,
  HelpText<"Put MODULE files in <dir>">,
  DocBrief<[{This option specifies where to put .mod files for compiled modules.
It is also added to the list of directories to be searched by an USE statement.
The default is the current directory.}]>;

def ffixed_form : Flag<["-"], "ffixed-form">, Group<f_Group>,
  HelpText<"Process source files in fixed form">;
def ffree_form : Flag<["-"], "ffree-form">, Group<f_Group>,
  HelpText<"Process source files in free form">;
def ffixed_line_length_EQ : Joined<["-"], "ffixed-line-length=">, Group<f_Group>,
  HelpText<"Use <value> as character line width in fixed mode">,
  DocBrief<[{Set column after which characters are ignored in typical fixed-form lines in the source
file}]>;
def ffixed_line_length_VALUE : Joined<["-"], "ffixed-line-length-">, Group<f_Group>, Alias<ffixed_line_length_EQ>;
def fconvert_EQ : Joined<["-"], "fconvert=">, Group<f_Group>,
  HelpText<"Set endian conversion of data for unformatted files">;
def fdefault_double_8 : Flag<["-"],"fdefault-double-8">, Group<f_Group>,
  HelpText<"Set the default double precision kind to an 8 byte wide type">;
def fdefault_integer_8 : Flag<["-"],"fdefault-integer-8">, Group<f_Group>,
  HelpText<"Set the default integer and logical kind to an 8 byte wide type">;
def fdefault_real_8 : Flag<["-"],"fdefault-real-8">, Group<f_Group>,
  HelpText<"Set the default real kind to an 8 byte wide type">;
def fdefault_integer_4 : Flag<["-"],"fdefault-integer-4">, Group<f_Group>,
  HelpText<"Set the default integer and logical kind to a 4 byte wide type">;
def fdefault_real_4 : Flag<["-"],"fdefault-real-4">, Group<f_Group>,
  HelpText<"Set the default real kind to a 4 byte wide type">;
def fdisable_real_3 : Flag<["-"],"fdisable-real-3">, Group<f_Group>,
   HelpText<"Disable real(KIND=3) from TargetCharacteristics">, Flags<[HelpHidden]>;
def fdisable_real_10 : Flag<["-"],"fdisable-real-10">, Group<f_Group>,
  HelpText<"Disable real(KIND=10) from TargetCharacteristics">, Flags<[HelpHidden]>;
def fdisable_integer_2 : Flag<["-"],"fdisable-integer-2">, Group<f_Group>,
  HelpText<"Disable integer(KIND=2) from TargetCharacteristics">, Flags<[HelpHidden]>;
def fdisable_integer_16 : Flag<["-"],"fdisable-integer-16">, Group<f_Group>,
  HelpText<"Disable integer(KIND=16) from TargetCharacteristics">, Flags<[HelpHidden]>;
def flarge_sizes : Flag<["-"],"flarge-sizes">, Group<f_Group>,
  HelpText<"Use INTEGER(KIND=8) for the result type in size-related intrinsics">;

def falternative_parameter_statement : Flag<["-"], "falternative-parameter-statement">, Group<f_Group>,
  HelpText<"Enable the old style PARAMETER statement">;
def fintrinsic_modules_path : Separate<["-"], "fintrinsic-modules-path">,  Group<f_Group>, MetaVarName<"<dir>">,
  HelpText<"Specify where to find the compiled intrinsic modules">,
  DocBrief<[{This option specifies the location of pre-compiled intrinsic modules,
  if they are not in the default location expected by the compiler.}]>;
def fintrinsic_modules_path_EQ : Joined<["-"], "fintrinsic-modules-path=">,
  Group<f_Group>, Alias<fintrinsic_modules_path>;

defm backslash : OptInFC1FFlag<"backslash", "Specify that backslash in string introduces an escape character">;
defm xor_operator : OptInFC1FFlag<"xor-operator", "Enable .XOR. as a synonym of .NEQV.">;
defm logical_abbreviations : OptInFC1FFlag<"logical-abbreviations", "Enable logical abbreviations">;
defm implicit_none : OptInFC1FFlag<"implicit-none", "No implicit typing allowed unless overridden by IMPLICIT statements">;
defm implicit_none_ext : OptInFC1FFlag<"implicit-none-ext", "No implicit externals allowed">;
defm underscoring : OptInFC1FFlag<"underscoring", "Appends one trailing underscore to external names">;
defm ppc_native_vec_elem_order: BoolOptionWithoutMarshalling<"f", "ppc-native-vector-element-order",
  PosFlag<SetTrue, [], [ClangOption], "Specifies PowerPC native vector element order (default)">,
  NegFlag<SetFalse, [], [ClangOption], "Specifies PowerPC non-native vector element order">>;
defm unsigned : OptInFC1FFlag<"unsigned", "Enables UNSIGNED type">;

def fno_automatic : Flag<["-"], "fno-automatic">, Group<f_Group>,
  HelpText<"Implies the SAVE attribute for non-automatic local objects in subprograms unless RECURSIVE">;

defm repack_arrays
    : BoolOptionWithoutMarshalling<
          "f", "repack-arrays", PosFlag<SetTrue, [], [], "Pack">,
          NegFlag<SetFalse, [], [], "Do not pack">,
          BothFlags<[], [],
                    " non-contiguous assumed shape dummy arrays into "
                    "contiguous memory">>,
      DocBrief<[{Create temporary copies of non-contiguous assumed shape dummy
arrays in subprogram prologues, and destroy them in subprogram epilogues.
The temporary copy is initialized with values from the original array
in the prologue, if needed. In the epilogue, the current values
in the temporary array are copied into the original array, if needed.

Accessing the contiguous temporary in the program code may result
in faster execution comparing to accessing elements of the original array,
when they are sparse in memory. At the same time, the overhead
of copying values between the original and the temporary arrays
may be significant, which may slow down some programs.

Enabling array repacking may also change the behavior of certain
programs:

* The copy actions may introduce a data race in valid OpenACC/OpenMP programs.
  For example, if different threads execute the same subprogram
  with a non-contiguous assumed shape dummy array, and the different threads
  access unrelated parts of the array, then the whole array copy
  made in each thread will cause a data race.
* OpenACC/OpenMP offload programs may behave incorrectly with regards
  to the device data environment, due to the fact that the original
  array and the temporary may have different presence status on the device.
* ``IS_CONTIGUOUS`` intrinsic may return ``TRUE`` with the array repacking
  enabled, whereas if would return ``FALSE`` with the repacking disabled.
* The result of ``LOC`` intrinsic applied to an actual argument associated
  with a non-contiguous assumed shape dummy array, may be different
  from the result of ``LOC`` applied to the dummy array.}]>;

def frepack_arrays_contiguity_EQ
    : Joined<["-"], "frepack-arrays-contiguity=">,
      Group<f_Group>,
      Values<"whole,innermost">,
      HelpText<
          "When -frepack-arrays is in effect, 'whole' enables "
          "repacking for arrays that are non-contiguous in any dimension, "
          "'innermost' enables repacking for arrays that are non-contiguous "
          "in the innermost dimension (the default)">;

defm save_main_program : BoolOptionWithoutMarshalling<"f", "save-main-program",
  PosFlag<SetTrue, [], [],
    "Place all main program variables in static memory (otherwise scalars may be placed on the stack)">,
  NegFlag<SetFalse, [], [],
    "Allow placing main program variables on the stack (default)">>;

defm stack_arrays : BoolOptionWithoutMarshalling<"f", "stack-arrays",
  PosFlag<SetTrue, [], [ClangOption], "Attempt to allocate array temporaries on the stack, no matter their size">,
  NegFlag<SetFalse, [], [ClangOption], "Allocate array temporaries on the heap (default)">>;

defm safe_trampoline : BoolOptionWithoutMarshalling<"f",
  "safe-trampoline",
  PosFlag<SetTrue, [], [FlangOption], "Use W^X compliant runtime trampoline pool for internal procedures">,
  NegFlag<SetFalse, [], [FlangOption], "Use stack-based trampolines for internal procedures (default, may require executable stack)">>;

defm loop_versioning : BoolOptionWithoutMarshalling<"f", "version-loops-for-stride",
  PosFlag<SetTrue, [], [ClangOption], "Create unit-strided versions of loops">,
   NegFlag<SetFalse, [], [ClangOption], "Do not create unit-strided loops (default)">>;

defm stack_repack_arrays
    : BoolOptionWithoutMarshalling<
          "f", "stack-repack-arrays",
          PosFlag<SetTrue, [], [],
                  "Attempt to allocate array temporaries created under "
                  "-frepack-arrays on the stack">,
          NegFlag<
              SetFalse, [], [],
              "Allocate -frepack-arrays temporaries on the heap (default)">>,
      DocBrief<[{Controls whether the array temporaries created under
**-frepack-arrays** are allocated on the stack or on the heap.

By default, the heap is used. Allocations of polymorphic types
are always done on the heap, though this may change in future releases.
  }]>;

defm unsafe_cray_pointers : BoolOptionWithoutMarshalling<"f", "unsafe-cray-pointers",
  PosFlag<SetTrue, [], [FlangOption, FC1Option], "Optimizations allow for unsafe Cray pointer usages">,
  NegFlag<SetFalse, [], [FlangOption, FC1Option], "Optimizations don't allow for unsafe Cray pointer usages (default)">>;

def relaxed_c_loc : Flag<["-"], "frelaxed-c-loc-checks">, Group<f_Group>,
  Visibility<[FlangOption, FC1Option]>,
  HelpText<"Allow a data object or function pointer as the C_LOC() argument">,
  DocBrief<[{
    Allow a data object or function pointer as the C_LOC() argument
    (not just pointers/targets). The compiler will not reason about aliases
    created through non-target non-pointer arguments and code generated
    using such aliases may exhibit unexpected behavior.}]>;

def fhermetic_module_files : Flag<["-"], "fhermetic-module-files">, Group<f_Group>,
  HelpText<"Emit hermetic module files (no nested USE association)">;

def fdo_concurrent_to_openmp_EQ : Joined<["-"], "fdo-concurrent-to-openmp=">,
  HelpText<"Try to map `do concurrent` loops to OpenMP [none|host|device]">,
      Values<"none, host, device">;

def J : JoinedOrSeparate<["-"], "J">,
  Flags<[RenderJoined]>, Group<gfortran_Group>, Alias<module_dir>;

def ffast_real_mod : Flag<["-"], "ffast-real-mod">, Group<f_Group>,
  HelpText<"Enable optimization of MOD for REAL types">;
def fno_fast_real_mod : Flag<["-"], "fno-fast-real-mod">, Group<f_Group>,
  HelpText<"Disable optimization of MOD for REAL types in presence of -ffast-math">;

defm init_global_zero : BoolOptionWithoutMarshalling<"f", "init-global-zero",
  PosFlag<SetTrue, [], [], "Zero initialize globals without default initialization (default)">,
  NegFlag<SetFalse, [], [], "Do not zero initialize globals without default initialization">>;

def fno_realloc_lhs : Flag<["-"], "fno-realloc-lhs">, Group<f_Group>,
  HelpText<"An allocatable left-hand side of an intrinsic assignment is assumed to be allocated and match the shape/type of the right-hand side">;
def frealloc_lhs : Flag<["-"], "frealloc-lhs">, Group<f_Group>,
  HelpText<"If an allocatable left-hand side of an intrinsic assignment is unallocated or its shape/type does not match the right-hand side, then it is automatically (re)allocated">;

//===----------------------------------------------------------------------===//
// Coarray Options
//===----------------------------------------------------------------------===//

def fcoarray : Flag<["-"], "fcoarray">,
               Group<f_Group>,
               HelpText<"Enable Coarray features">;

} // let Visibility = [FC1Option, FlangOption]

//===----------------------------------------------------------------------===//
// FC1 Options
//===----------------------------------------------------------------------===//

let Visibility = [FC1Option] in {

def ffp_maxmin_behavior_EQ
    : Joined<["-"], "ffp-maxmin-behavior=">,
      Flags<[HelpHidden]>,
      Group<f_Group>,
      Values<"legacy,portable,extremum,extremenum">,
      HelpText<"Control max/min and [max|min][loc|val] behavior: "
               "legacy (cmp+select), portable (same as legacy, "
               " but may use max/minNum when -fno-signed-zeros "
               "-fno-honor-nans), extremum (IEEE-754-2019 maximum/minimum), "
               "extremenum (IEEE-754-2008 max/minNum)">;

def fget_definition : MultiArg<["-"], "fget-definition", 3>,
  HelpText<"Get the symbol definition from <line> <start-column> <end-column>">,
  Group<Action_Group>;
def test_io : Flag<["-"], "test-io">, Group<Action_Group>,
  HelpText<"Run the InputOuputTest action. Use for development and testing only.">;
def fdebug_unparse_no_sema : Flag<["-"], "fdebug-unparse-no-sema">, Group<Action_Group>,
  HelpText<"Unparse and stop (skips the semantic checks)">,
  DocBrief<[{Only run the parser, then unparse the parse-tree and output the
generated Fortran source file. Semantic checks are disabled.}]>;
def fdebug_unparse : Flag<["-"], "fdebug-unparse">, Group<Action_Group>,
  HelpText<"Unparse and stop.">,
  DocBrief<[{Run the parser and the semantic checks. Then unparse the
parse-tree and output the generated Fortran source file.}]>;
def fdebug_unparse_with_symbols : Flag<["-"], "fdebug-unparse-with-symbols">, Group<Action_Group>,
  HelpText<"Unparse with symbols and stop.">;
def fdebug_unparse_with_modules : Flag<["-"], "fdebug-unparse-with-modules">, Group<Action_Group>,
  HelpText<"Unparse with dependent modules and stop.">;
def fdebug_dump_symbols : Flag<["-"], "fdebug-dump-symbols">, Group<Action_Group>,
  HelpText<"Dump symbols after the semantic analysis">;
def fdebug_dump_parse_tree : Flag<["-"], "fdebug-dump-parse-tree">, Group<Action_Group>,
  HelpText<"Dump the parse tree">,
  DocBrief<[{Run the Parser and the semantic checks, and then output the
parse tree.}]>;
def fdebug_dump_pft : Flag<["-"], "fdebug-dump-pft">, Group<Action_Group>,
  HelpText<"Dump the pre-fir parse tree">;
def fdebug_dump_parse_tree_no_sema : Flag<["-"], "fdebug-dump-parse-tree-no-sema">, Group<Action_Group>,
  HelpText<"Dump the parse tree (skips the semantic checks)">,
  DocBrief<[{Run the Parser and then output the parse tree. Semantic
checks are disabled.}]>;
def fdebug_dump_all : Flag<["-"], "fdebug-dump-all">, Group<Action_Group>,
  HelpText<"Dump symbols and the parse tree after the semantic checks">;
def fdebug_dump_provenance : Flag<["-"], "fdebug-dump-provenance">, Group<Action_Group>,
  HelpText<"Dump provenance">;
def fdebug_dump_parsing_log : Flag<["-"], "fdebug-dump-parsing-log">, Group<Action_Group>,
  HelpText<"Run instrumented parse and dump the parsing log">;
def fdebug_measure_parse_tree : Flag<["-"], "fdebug-measure-parse-tree">, Group<Action_Group>,
  HelpText<"Measure the parse tree">;
def fdebug_pre_fir_tree : Flag<["-"], "fdebug-pre-fir-tree">, Group<Action_Group>,
  HelpText<"Dump the pre-FIR tree">;
def fdebug_module_writer : Flag<["-"],"fdebug-module-writer">,
  HelpText<"Enable debug messages while writing module files">;
def fget_symbols_sources : Flag<["-"], "fget-symbols-sources">, Group<Action_Group>,
  HelpText<"Dump symbols and their source code locations">;

def module_suffix : Separate<["-"], "module-suffix">,  Group<f_Group>, MetaVarName<"<suffix>">,
  HelpText<"Use <suffix> as the suffix for module files (the default value is `.mod`)">;
def fno_reformat : Flag<["-"], "fno-reformat">, Group<Preprocessor_Group>,
  HelpText<"Dump the cooked character stream in -E mode">;
def fpreprocess_include_lines : Flag<["-"], "fpreprocess-include-lines">, Group<Preprocessor_Group>,
  HelpText<"Treat INCLUDE lines like #include directives in -E mode">;
defm analyzed_objects_for_unparse : OptOutFC1FFlag<"analyzed-objects-for-unparse", "", "Do not use the analyzed objects when unparsing">;

def emit_fir : Flag<["-"], "emit-fir">, Group<Action_Group>,
  HelpText<"Build the parse tree, then lower it to FIR">;
def emit_mlir : Flag<["-"], "emit-mlir">, Alias<emit_fir>;

def emit_hlfir : Flag<["-"], "emit-hlfir">, Group<Action_Group>,
  HelpText<"Build the parse tree, then lower it to HLFIR">;

} // let Visibility = [FC1Option]
