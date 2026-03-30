//==--- DiagnosticCrossTUKinds.td - Cross Translation Unit diagnostics ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

let Component = "CrossTU" in {

def err_ctu_error_opening : Error<
  "error opening '%0': required by the CrossTU functionality">;

def err_extdefmap_parsing : Error<
  "error parsing index file: '%0' line: %1 '<USR-Length>:<USR> <File-Path>' "
  "format expected">;

def err_invlist_parsing : Error<
  "error parsing invocation list file: '%0' line: %1 "
  "'<source-file>: [<compiler>, <arg1>, ...]' YAML mapping format expected">;

def err_multiple_def_index : Error<
  "multiple definitions are found for the same key in index">;

def warn_multiple_entries_invlist : Warning<
  "multiple invocations for '%0' are found in the invocation list">,
  InGroup<CrossTU>;

def warn_invlist_missing_file : Warning<
  "invocation for '%0' is missing in the invocation list">, InGroup<CrossTU>;

def warn_ctu_incompat_triple : Warning<
  "imported AST from '%0' had been generated for a different target, "
  "current: %1, imported: %2">, InGroup<CrossTU>;

def warn_ctu_import_failure : Warning<
  "import of an external symbol for CTU failed: %0">, InGroup<CrossTU>;

def err_ctu_import_failure: Error<
  "import of an external symbol for CTU failed: %0">;

def remark_ctu_import_threshold_reached: Remark<
  "reached a the CTU-import threshold before trying to import definition">,
  InGroup<CrossTU>;

def warn_ctu_incompat_lang : Warning<
  "imported AST from '%0' had been generated for a different language, "
  "current: %1, imported: %2">, InGroup<CrossTU>;
}
