//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def CodeGenCLOptions : OptionsStruct<"clang::CodeGen">;

// BackendUtil.cpp

// Experiment to mark cold functions as optsize/minsize/optnone.
// TODO: remove once this is exposed as a proper driver flag.
defm : EnumField<"pgo-cold-func-opt", "llvm::PGOOptions::ColdFuncOpt",
  "llvm::PGOOptions::ColdFuncOpt::Default", "default,optsize,minsize,optnone",
  ["llvm::PGOOptions::ColdFuncOpt::Default",
   "llvm::PGOOptions::ColdFuncOpt::OptSize",
   "llvm::PGOOptions::ColdFuncOpt::MinSize",
   "llvm::PGOOptions::ColdFuncOpt::OptNone"],
  "Function attribute to apply to cold functions as determined by PGO">;
// Experiment to move sanitizers earlier.
defm : BoolField<"sanitizer-early-opt-ep", "false",
  "Insert sanitizers on OptimizerEarlyEP">;

// CGExpr.cpp

// TODO: consider deprecating ubsan-guard-checks; functionality is subsumed by
// -fsanitize-skip-hot-cutoff
defm : BoolField<"ubsan-guard-checks", "false",
  "Guard UBSAN checks with `llvm.allow.ubsan.check()`">;

// CodeGenModule.cpp

defm : BoolField<"limited-coverage-experimental", "false",
  "Emit limited coverage mapping information (experimental)">;

// CodeGenPGO.cpp

defm : BoolField<"value-profiling", "false",
  "Enable value profiling">;
// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"enable-value-profiling", "value-profiling">;

// CoverageMappingGen.cpp

defm : BoolField<"emptyline-comment-coverage", "true",
  "Emit emptylines and comment lines as skipped regions (only disable it on test)">;
defm : BoolField<"single-byte-coverage", "false",
  "Enable single byte coverage">;
defm : BoolField<"system-headers-coverage", "false",
  "Enable collecting coverage from system headers">;
// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"enable-single-byte-coverage", "single-byte-coverage">;
