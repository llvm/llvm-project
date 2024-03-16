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
} // end of command line category.

} // end of InstallAPI component
