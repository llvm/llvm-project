//===- SSAFLinker.cpp - SSAF Linker ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements the SSAF entity linker tool. Its default behavior is to
//  link N inputs (TU summaries, static libraries, and multi-arch static
//  libraries) into one LU summary via the EntityLinker framework. It also
//  provides the `static-library` subcommand for bundling TU summaries into a
//  StaticLibrary, and the `multi-arch` subcommand for bundling StaticLibrary
//  and SharedLibrary members (or existing multi-arch bundles) into
//  MultiArchStaticLibrary or MultiArchSharedLibrary.
//
//===----------------------------------------------------------------------===//

#include "LinkCLI.h"
#include "MultiArchCreateCLI.h"
#include "StaticLibraryCreateCLI.h"

#include "clang/ScalableStaticAnalysis/SSAFForceLinker.h" // IWYU pragma: keep
#include "clang/ScalableStaticAnalysis/Tool/Utils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace llvm;
using namespace clang::ssaf;

namespace {

//===----------------------------------------------------------------------===//
// Command-Line Options
//===----------------------------------------------------------------------===//

cl::OptionCategory SsafLinkerCategory("clang-ssaf-linker options");

// The `static-library` subcommand groups all StaticLibrary operations.
cl::SubCommand StaticLibraryCmd("static-library",
                                "Operations on StaticLibraries");

// The `multi-arch` subcommand groups all multi-architecture operations.
cl::SubCommand MultiArchCmd("multi-arch",
                            "Operations on multi-architecture StaticLibrary "
                            "and SharedLibrary artifacts");

// Top-level (default) `link` action positionals.
cl::list<std::string> InputPaths(cl::Positional, cl::desc("<input files>"),
                                 cl::OneOrMore, cl::cat(SsafLinkerCategory));

cl::opt<std::string> OutputPath("o", cl::desc("Output file path"),
                                cl::value_desc("path"), cl::Required,
                                cl::cat(SsafLinkerCategory));

cl::opt<std::string> TargetTriple(
    "target-triple",
    cl::desc(
        "Target triple of the link unit (defaults to the first input's; "
        "required when the first input is a multi-arch static library with "
        "several members)"),
    cl::value_desc("triple"), cl::cat(SsafLinkerCategory));

// --verbose and --time apply to every subcommand.
cl::opt<bool> Verbose("verbose", cl::desc("Enable verbose output"),
                      cl::init(false), cl::cat(SsafLinkerCategory),
                      cl::sub(cl::SubCommand::getTopLevel()),
                      cl::sub(StaticLibraryCmd), cl::sub(MultiArchCmd));

cl::opt<bool> Time("time", cl::desc("Enable timing"), cl::init(false),
                   cl::cat(SsafLinkerCategory),
                   cl::sub(cl::SubCommand::getTopLevel()),
                   cl::sub(StaticLibraryCmd), cl::sub(MultiArchCmd));

// The `static-library` subcommand's verb positional. Declared BEFORE
// StaticLibraryInputs so cl-lib binds argv[0] under the subcommand to the
// verb rather than to the greedy input list.
cl::opt<std::string> StaticLibraryVerb(cl::Positional, cl::Required,
                                       cl::sub(StaticLibraryCmd),
                                       cl::desc("<verb>"),
                                       cl::cat(SsafLinkerCategory));

// The `static-library` subcommand's action-specific positional input
// list. Currently consumed by `static-library create`; if future verbs
// need different input shapes they'll declare their own positionals.
cl::list<std::string> StaticLibraryInputs(cl::Positional,
                                          cl::sub(StaticLibraryCmd),
                                          cl::desc("<TU summary files>"),
                                          cl::cat(SsafLinkerCategory));

cl::opt<std::string> StaticLibraryOutput("o", cl::Required,
                                         cl::sub(StaticLibraryCmd),
                                         cl::desc("Output file path"),
                                         cl::value_desc("path"),
                                         cl::cat(SsafLinkerCategory));

cl::opt<std::string> StaticLibraryNamespace(
    "namespace", cl::sub(StaticLibraryCmd),
    cl::desc("Namespace name for the StaticLibrary (defaults to output "
             "file stem)"),
    cl::value_desc("name"), cl::cat(SsafLinkerCategory));

cl::opt<std::string> StaticLibraryTriple(
    "target-triple", cl::sub(StaticLibraryCmd),
    cl::desc("Target triple (defaults to inputs' triple; must match all "
             "inputs when set)"),
    cl::value_desc("triple"), cl::cat(SsafLinkerCategory));

// The `multi-arch` subcommand's verb positional. Declared BEFORE
// MultiArchInputs so cl-lib binds argv[0] under the subcommand to the verb
// rather than to the greedy input list.
cl::opt<std::string> MultiArchVerb(cl::Positional, cl::Required,
                                   cl::sub(MultiArchCmd), cl::desc("<verb>"),
                                   cl::cat(SsafLinkerCategory));

// The `multi-arch` subcommand's action-specific positional input list.
// Currently consumed by `multi-arch create`.
cl::list<std::string>
    MultiArchInputs(cl::Positional, cl::sub(MultiArchCmd),
                    cl::desc("<static-library or shared-library files>"),
                    cl::cat(SsafLinkerCategory));

cl::opt<std::string> MultiArchOutput("o", cl::Required, cl::sub(MultiArchCmd),
                                     cl::desc("Output file path"),
                                     cl::value_desc("path"),
                                     cl::cat(SsafLinkerCategory));

//===----------------------------------------------------------------------===//
// StaticLibrary Verbs
//===----------------------------------------------------------------------===//

// Verb strings for the `static-library` subcommand. Kept in sync with
// UnknownStaticLibraryVerb below.
constexpr const char *StaticLibraryCreateVerb = "create";

//===----------------------------------------------------------------------===//
// MultiArch Verbs
//===----------------------------------------------------------------------===//

// Verb strings for the `multi-arch` subcommand. Kept in sync with
// UnknownMultiArchVerb below.
constexpr const char *MultiArchCreateVerb = "create";

//===----------------------------------------------------------------------===//
// Error Messages
//===----------------------------------------------------------------------===//

namespace LocalErrorMessages {

constexpr const char *UnknownStaticLibraryVerb =
    "unknown static-library verb '{0}': expected 'create'";

constexpr const char *UnknownMultiArchVerb =
    "unknown multi-arch verb '{0}': expected 'create'";

} // namespace LocalErrorMessages

//===----------------------------------------------------------------------===//
// default (no subcommand) link action
//===----------------------------------------------------------------------===//

void runLink(llvm::TimerGroup &TG) {
  LinkCLI LC;
  LC.run(TG, InputPaths, OutputPath, TargetTriple, Verbose, Time);
}

//===----------------------------------------------------------------------===//
// static-library subcommand dispatch
//===----------------------------------------------------------------------===//

void runStaticLibrary(llvm::TimerGroup &TG) {
  if (StaticLibraryVerb == StaticLibraryCreateVerb) {
    StaticLibraryCreateCLI::Config Cfg;
    Cfg.InputPaths = StaticLibraryInputs;
    Cfg.OutputPath = StaticLibraryOutput;
    Cfg.Namespace = StaticLibraryNamespace;
    Cfg.TargetTriple = StaticLibraryTriple;
    Cfg.Verbose = Verbose;
    Cfg.Time = Time;

    StaticLibraryCreateCLI SLC;
    SLC.run(TG, Cfg);
    return;
  }
  fail(LocalErrorMessages::UnknownStaticLibraryVerb,
       StaticLibraryVerb.getValue());
}

//===----------------------------------------------------------------------===//
// multi-arch subcommand dispatch
//===----------------------------------------------------------------------===//

void runMultiArch(llvm::TimerGroup &TG) {
  if (MultiArchVerb == MultiArchCreateVerb) {
    MultiArchCreateCLI MAC;
    MAC.run(TG, MultiArchInputs, MultiArchOutput, Verbose, Time);
    return;
  }
  fail(LocalErrorMessages::UnknownMultiArchVerb, MultiArchVerb.getValue());
}

} // namespace

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

int main(int argc, const char **argv) {
  llvm::StringRef ToolHeading = "SSAF Linker";

  InitLLVM X(argc, argv);
  initTool(argc, argv, "0.1", SsafLinkerCategory, ToolHeading);

  llvm::TimerGroup Timers(getToolName(), ToolHeading);

  if (StaticLibraryCmd) {
    runStaticLibrary(Timers);
  } else if (MultiArchCmd) {
    runMultiArch(Timers);
  } else {
    // Default (no subcommand): run the linker pipeline.
    runLink(Timers);
  }

  return 0;
}
