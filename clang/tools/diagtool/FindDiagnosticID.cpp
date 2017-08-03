//===- FindDiagnosticID.cpp - diagtool tool for finding diagnostic id -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DiagTool.h"
#include "DiagnosticNames.h"
#include "clang/Basic/AllDiagnostics.h"
#include "llvm/Support/CommandLine.h"

DEF_DIAGTOOL("find-diagnostic-id", "Print the id of the given diagnostic",
             FindDiagnosticID)

using namespace clang;
using namespace diagtool;

static Optional<DiagnosticRecord>
findDiagnostic(ArrayRef<DiagnosticRecord> Diagnostics, StringRef Name) {
  for (const auto &Diag : Diagnostics) {
    StringRef DiagName = Diag.getName();
    if (DiagName == Name)
      return Diag;
  }
  return None;
}

int FindDiagnosticID::run(unsigned int argc, char **argv,
                          llvm::raw_ostream &OS) {
  static llvm::cl::OptionCategory FindDiagnosticIDOptions(
      "diagtool find-diagnostic-id options");

  static llvm::cl::opt<std::string> DiagnosticName(
      llvm::cl::Positional, llvm::cl::desc("<diagnostic-name>"),
      llvm::cl::Required, llvm::cl::cat(FindDiagnosticIDOptions));

  std::vector<const char *> Args;
  Args.push_back("find-diagnostic-id");
  for (const char *A : llvm::makeArrayRef(argv, argc))
    Args.push_back(A);

  llvm::cl::HideUnrelatedOptions(FindDiagnosticIDOptions);
  llvm::cl::ParseCommandLineOptions((int)Args.size(), Args.data(),
                                    "Diagnostic ID mapping utility");

  ArrayRef<DiagnosticRecord> AllDiagnostics = getBuiltinDiagnosticsByName();
  Optional<DiagnosticRecord> Diag =
      findDiagnostic(AllDiagnostics, DiagnosticName);
  if (!Diag) {
    llvm::errs() << "error: invalid diagnostic '" << DiagnosticName << "'\n";
    return 1;
  }
  OS << Diag->DiagID << "\n";
  return 0;
}
