//===--- InstallAPIOpts.td ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the specific options for InstallAPI.
//
//===----------------------------------------------------------------------===//

// Include the common option parsing interfaces.
include "llvm/Option/OptParser.td"


/////////
// Options

// TextAPI options.
def filetype : Joined<["--"], "filetype=">,
  HelpText<"Specify the output file type (tbd-v4 or tbd-v5)">;

// Verification options.
def verify_against : Separate<["-"], "verify-against">,
  HelpText<"Verify the specified dynamic library/framework against the headers">;
def verify_against_EQ : Joined<["--"], "verify-against=">, Alias<verify_against>;
def verify_mode_EQ : Joined<["--"], "verify-mode=">,
  HelpText<"Specify the severity and extend of the validation. Valid modes are ErrorsOnly, ErrorsAndWarnings, and Pedantic.">;
def demangle : Flag<["--", "-"], "demangle">,
  HelpText<"Demangle symbols when printing warnings and errors">;
