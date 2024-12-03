//===---- ClangReadDiagnostics.cpp - clang-read-diagnostics tool -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tool is for reading clang diagnostics files from -serialize-diagnostics.
//
// Example usage:
//
// $ clang -serialize-diagnostics foo.c.diag foo.c
// $ clang-read-diagnostics foo.c.diag
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/SerializedDiagnosticReader.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/LineEditor/LineEditor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"

#include <optional>
#include <string>

using namespace clang;
using namespace clang::serialized_diags;
using namespace llvm;

static cl::list<std::string> InputFiles(cl::Sink, cl::desc("<input files...>"), cl::Required);

class BasicSerializedDiagnosticReader : public SerializedDiagnosticReader {
public:
  struct DecodedDiagnostics {
    unsigned Severity;
    StringRef Filename;
    unsigned Line;
    unsigned Col;
    StringRef Category;
    StringRef Flag;
    StringRef Message;
  };

  SmallVector<DecodedDiagnostics> getDiagnostics() {
    SmallVector<DecodedDiagnostics> Ret;
    for (const auto &Diag : Diagnostics_) {
      auto Filename = FilenameIdx_.at(Diag.Location.FileID);
      auto Category = CategoryIdx_.at(Diag.Category);
      auto Flag = FlagIdx_.at(Diag.Flag);
      Ret.emplace_back(DecodedDiagnostics{Diag.Severity, Filename,
                                          Diag.Location.Line, Diag.Location.Col,
                                          Category, Flag, Diag.Message});
    }
    return Ret;
  }

  void dump() {
    for (const auto &Diag : getDiagnostics()) {
      llvm::dbgs() << Diag.Filename << ":" << Diag.Line << ":" << Diag.Col
                   << ": " << Diag.Message << " [Category=\'" << Diag.Category
                   << "', flag=" << Diag.Flag << "]" << "\n";
    }
  }

protected:
  virtual std::error_code visitCategoryRecord(unsigned ID,
                                              StringRef Name) override {
    const auto &[_, Inserted] = CategoryIdx_.try_emplace(ID, Name);
    assert(Inserted && "duplicate IDs");
    return {};
  }

  virtual std::error_code visitDiagFlagRecord(unsigned ID,
                                              StringRef Name) override {
    const auto &[_, Inserted] = FlagIdx_.try_emplace(ID, Name);
    assert(Inserted && "duplicate IDs");
    return {};
  }

  virtual std::error_code visitFilenameRecord(unsigned ID, unsigned Size,
                                              unsigned Timestamp,
                                              StringRef Name) override {
    const auto &[_, Inserted] = FilenameIdx_.try_emplace(ID, Name);
    assert(Inserted && "duplicate IDs");
    return {};
  }

  virtual std::error_code
  visitDiagnosticRecord(unsigned Severity, const Location &Location,
                        unsigned Category, unsigned Flag, StringRef Message) override {
    Diagnostics_.emplace_back(
        RawDiagnostic{Severity, Location, Category, Flag, Message});
    return {};
  }

private:
  struct RawDiagnostic {
    unsigned Severity;
    Location Location;
    unsigned Category;
    unsigned Flag;
    StringRef Message;
  };

  DenseMap<unsigned, StringRef> CategoryIdx_;
  DenseMap<unsigned, StringRef> FlagIdx_;
  DenseMap<unsigned, StringRef> FilenameIdx_;
  std::vector<RawDiagnostic> Diagnostics_;
};

int main(int argc, const char **argv) {
  cl::ParseCommandLineOptions(argc, argv);
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  for (const auto &File : InputFiles) {
    BasicSerializedDiagnosticReader BSDR{};
    BSDR.readDiagnostics(File);
    BSDR.dump();
  }

  return 0;
}
