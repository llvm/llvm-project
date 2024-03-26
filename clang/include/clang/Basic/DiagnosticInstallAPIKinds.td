//==--- DiagnosticInstallAPIKinds.td - installapi diagnostics -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// InstallAPI Diagnostics
//===----------------------------------------------------------------------===//

let Component = "InstallAPI" in {
let CategoryName = "Command line" in {
def err_cannot_write_file : Error<"cannot write file '%0': %1">;
def err_no_install_name : Error<"no install name specified: add -install_name <path>">;
def err_no_output_file: Error<"no output file specified">;
def err_no_such_header_file : Error<"no such %select{public|private|project}1 header file: '%0'">;
def warn_no_such_excluded_header_file : Warning<"no such excluded %select{public|private}0 header file: '%1'">, InGroup<InstallAPIViolation>;
def warn_glob_did_not_match: Warning<"glob '%0' did not match any header file">, InGroup<InstallAPIViolation>;
} // end of command line category.

let CategoryName = "Verification" in {
def warn_target: Warning<"violations found for %0">, InGroup<InstallAPIViolation>;
def err_library_missing_symbol : Error<"declaration has external linkage, but dynamic library doesn't have symbol '%0'">;
def warn_library_missing_symbol : Warning<"declaration has external linkage, but dynamic library doesn't have symbol '%0'">, InGroup<InstallAPIViolation>;
def err_library_hidden_symbol : Error<"declaration has external linkage, but symbol has internal linkage in dynamic library '%0'">;
def warn_library_hidden_symbol : Warning<"declaration has external linkage, but symbol has internal linkage in dynamic library '%0'">, InGroup<InstallAPIViolation>;
def warn_header_hidden_symbol : Warning<"symbol exported in dynamic library, but marked hidden in declaration '%0'">, InGroup<InstallAPIViolation>;
def err_header_hidden_symbol : Error<"symbol exported in dynamic library, but marked hidden in declaration '%0'">;
def err_header_symbol_missing : Error<"no declaration found for exported symbol '%0' in dynamic library">;
def warn_header_symbol_missing : Warning<"no declaration was found for exported symbol '%0' in dynamic library">, InGroup<InstallAPIViolation>;
def warn_header_availability_mismatch : Warning<"declaration '%0' is marked %select{available|unavailable}1,"
  " but symbol is %select{not |}2exported in dynamic library">, InGroup<InstallAPIViolation>;
def err_header_availability_mismatch : Error<"declaration '%0' is marked %select{available|unavailable}1,"
  " but symbol is %select{not |}2exported in dynamic library">;
def warn_dylib_symbol_flags_mismatch : Warning<"dynamic library symbol '%0' is "
  "%select{weak defined|thread local}1, but its declaration is not">, InGroup<InstallAPIViolation>;
def warn_header_symbol_flags_mismatch : Warning<"declaration '%0' is "
  "%select{weak defined|thread local}1, but symbol is not in dynamic library">, InGroup<InstallAPIViolation>;
def err_dylib_symbol_flags_mismatch : Error<"dynamic library symbol '%0' is "
  "%select{weak defined|thread local}1, but its declaration is not">;
def err_header_symbol_flags_mismatch : Error<"declaration '%0' is "
  "%select{weak defined|thread local}1, but symbol is not in dynamic library">;
} // end of Verification category.

} // end of InstallAPI component
