//===- FindDiagnosticID.cpp - diagtool tool for finding diagnostic id -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DiagTool.h"
#include "DiagnosticNames.h"
#include "clang/Basic/AllDiagnostics.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include <optional>

DEF_DIAGTOOL("find-diagnostic-id", "Print the id of the given diagnostic",
             FindDiagnosticID)

using namespace clang;
using namespace diagtool;

namespace {
using namespace llvm::opt;
enum ID {
  OPT_INVALID = 0,
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

#define OPTTABLE_CODE
#include "Opts.inc"

class FindDiagnosticIDOptTable : public OptTable {
public:
  FindDiagnosticIDOptTable() : OptTable(optionTables()) {
    setDashDashParsing(true);
  }
};
} // namespace

static StringRef getNameFromID(StringRef Name) {
  int DiagID;
  if(!Name.getAsInteger(0, DiagID)) {
    const DiagnosticRecord &Diag = getDiagnosticForID(DiagID);
    return Diag.getName();
  }
  return StringRef();
}

static std::optional<DiagnosticRecord>
findDiagnostic(ArrayRef<DiagnosticRecord> Diagnostics, StringRef Name) {
  for (const auto &Diag : Diagnostics) {
    StringRef DiagName = Diag.getName();
    if (DiagName == Name)
      return Diag;
  }
  return std::nullopt;
}

int FindDiagnosticID::run(unsigned int argc, char **argv,
                          llvm::raw_ostream &OS) {
  std::vector<const char *> Args;
  Args.push_back("diagtool find-diagnostic-id");
  for (const char *A : llvm::ArrayRef(argv, argc))
    Args.push_back(A);

  llvm::BumpPtrAllocator Alloc;
  llvm::StringSaver Saver(Alloc);
  FindDiagnosticIDOptTable Tbl;
  llvm::opt::InputArgList InputArgs =
      Tbl.parseArgs(Args.size(), const_cast<char **>(Args.data()), OPT_UNKNOWN,
                    Saver, [](StringRef Msg) {
                      llvm::errs() << "error: " << Msg << '\n';
                      exit(1);
                    });
  if (InputArgs.hasArg(OPT_help)) {
    Tbl.printHelp(OS, "diagtool find-diagnostic-id [options] <diagnostic-name>",
                  "Diagnostic ID mapping utility");
    return 0;
  }
  std::vector<std::string> Inputs = InputArgs.getAllArgValues(OPT_INPUT);
  if (Inputs.size() != 1) {
    llvm::errs() << "error: expected exactly one <diagnostic-name>\n";
    return 1;
  }
  StringRef DiagnosticName = Inputs[0];

  ArrayRef<DiagnosticRecord> AllDiagnostics = getBuiltinDiagnosticsByName();
  std::optional<DiagnosticRecord> Diag =
      findDiagnostic(AllDiagnostics, DiagnosticName);
  if (!Diag) {
    // Name to id failed, so try id to name.
    auto Name = getNameFromID(DiagnosticName);
    if (!Name.empty()) {
      OS << Name << '\n';
      return 0;
    }

    llvm::errs() << "error: invalid diagnostic '" << DiagnosticName << "'\n";
    return 1;
  }
  OS << Diag->DiagID << "\n";
  return 0;
}
