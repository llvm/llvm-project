//==--- DiagnosticSerializationKinds.td - serialization diagnostics -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

let Component = "Serialization" in {

def err_fe_unable_to_read_pch_file : Error<
    "unable to read PCH file %0: '%1'">;
def err_fe_not_a_pch_file : Error<
    "input is not a PCH file: '%0'">;
def err_fe_pch_malformed : Error<
    "malformed or corrupted PCH file: '%0'">, DefaultFatal;
def err_fe_pch_malformed_block : Error<
    "malformed block record in PCH file: '%0'">, DefaultFatal;
def err_fe_pch_error_at_end_block : Error<
    "error at end of module block in PCH file: '%0'">, DefaultFatal;
def err_fe_pch_file_modified : Error<
    "file '%0' has been modified since the precompiled header was built">,
    DefaultFatal;
def err_fe_pch_file_overridden : Error<
    "file '%0' from the precompiled header has been overridden">;

def err_pch_targetopt_mismatch : Error<
    "PCH file was compiled for the %0 '%1' but the current translation "
    "unit is being compiled for target '%2'">;
def err_pch_targetopt_feature_mismatch : Error<
    "%select{AST file|current translation unit}0 was compiled with the target "
    "feature'%1' but the %select{current translation unit is|AST file was}0 "
    "not">;
def err_pch_langopt_mismatch : Error<"%0 was %select{disabled|enabled}1 in "
    "PCH file but is currently %select{disabled|enabled}2">;
def err_pch_langopt_value_mismatch : Error<
  "%0 differs in PCH file vs. current file">;
  
def warn_pch_version_too_old : Error<
    "PCH file uses an older PCH format that is no longer supported">;
def warn_pch_version_too_new : Error<
    "PCH file uses a newer PCH format that cannot be read">;
def warn_pch_different_branch : Error<
    "PCH file built from a different branch (%0) than the compiler (%1)">;
def err_pch_with_compiler_errors : Error<
    "PCH file contains compiler errors">;
    
    
def err_pch_macro_def_undef : Error<
    "macro '%0' was %select{defined|undef'd}1 in the precompiled header but "
    "%select{undef'd|defined}1 on the command line">;
def err_pch_macro_def_conflict : Error<
    "definition of macro '%0' differs between the precompiled header ('%1') "
    "and the command line ('%2')">;
def err_pch_include_opt_missing : Error<
    "precompiled header depends on '%select{-include|-imacros}0 %1' option "
    "that is missing from the command line">;
def err_pch_include_opt_conflict : Error<
    "precompiled header option '%select{-include|-imacros}0 %1' conflicts with "
    "corresponding option '%select{-include|-imacros}0 %2' on command line">;
def err_pch_undef : Error<
    "%select{command line contains|precompiled header was built with}0 "
    "'-undef' but %select{precompiled header was not built with it|"
    "it is not present on the command line}0">;

def err_not_a_pch_file : Error<
    "'%0' does not appear to be a precompiled header file">, DefaultFatal;
}
