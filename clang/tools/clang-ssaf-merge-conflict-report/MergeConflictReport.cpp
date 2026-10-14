//===- MergeConflictReport.cpp - SSAF Conflict Report Merge Tool ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements the tool that merges SARIF conflict reports, as
//  written by clang-ssaf-src-edit-merge --sarif-conflicts-out or by this tool,
//  into one report. A conflict that several inputs report, such as one found
//  while merging the edits of two archives that are linked into the same
//  program, is kept once.
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Tool/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <set>
#include <string>
#include <tuple>
#include <vector>

using namespace llvm;
using namespace clang::ssaf;

namespace {

//===----------------------------------------------------------------------===//
// Command-Line Options
//===----------------------------------------------------------------------===//

cl::OptionCategory
    MergeConflictReportCategory("clang-ssaf-merge-conflict-report options");

cl::list<std::string> InputPaths(cl::Positional, cl::desc("<input.sarif>..."),
                                 cl::OneOrMore,
                                 cl::cat(MergeConflictReportCategory));

cl::opt<std::string> OutputPath("o", cl::desc("Output file path"),
                                cl::value_desc("path"), cl::Required,
                                cl::cat(MergeConflictReportCategory));

//===----------------------------------------------------------------------===//
// Error Messages
//===----------------------------------------------------------------------===//

namespace ErrorMessages {

constexpr const char *CannotRead = "failed to read '{0}': {1}";

constexpr const char *CannotParse = "failed to parse '{0}': {1}";

constexpr const char *NotOneRun =
    "'{0}' is not a SARIF log with exactly one run";

constexpr const char *OtherTool = "'{0}' was written by '{1}', not by '{2}'";

constexpr const char *CannotWrite = "failed to write '{0}': {1}";

} // namespace ErrorMessages

//===----------------------------------------------------------------------===//
// Reading and Writing
//===----------------------------------------------------------------------===//

json::Value readLog(StringRef Path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer = MemoryBuffer::getFile(Path);
  if (!Buffer) {
    fail(ErrorMessages::CannotRead, Path, Buffer.getError().message());
  }
  Expected<json::Value> Log = json::parse((*Buffer)->getBuffer());
  if (!Log) {
    fail(ErrorMessages::CannotParse, Path, toString(Log.takeError()));
  }
  return std::move(*Log);
}

json::Object &getRun(json::Value &Log, StringRef Path) {
  json::Object *Root = Log.getAsObject();
  json::Array *Runs = Root ? Root->getArray("runs") : nullptr;
  if (!Runs || Runs->size() != 1 || !Runs->front().getAsObject()) {
    fail(ErrorMessages::NotOneRun, Path);
  }
  return *Runs->front().getAsObject();
}

void writeLog(const json::Value &Log, StringRef Path) {
  std::error_code EC;
  ToolOutputFile Out(Path, EC, sys::fs::OF_Text);
  if (EC) {
    fail(ErrorMessages::CannotWrite, Path, EC.message());
  }
  Out.os() << formatv("{0:2}", Log) << "\n";
  Out.os().flush();
  if (Out.os().has_error()) {
    fail(ErrorMessages::CannotWrite, Path, Out.os().error().message());
  }
  Out.keep();
}

StringRef getString(const json::Object *Object, StringRef Key) {
  if (!Object) {
    return "";
  }
  return Object->getString(Key).value_or("");
}

/// Returns a copy of \p Object without the members named in \p Excluded.
json::Object copyExcept(const json::Object &Object,
                        ArrayRef<StringRef> Excluded) {
  json::Object Copy;
  for (const auto &[Key, Value] : Object) {
    if (!is_contained(Excluded, StringRef(Key))) {
      Copy[Key] = Value;
    }
  }
  return Copy;
}

//===----------------------------------------------------------------------===//
// Merging
//===----------------------------------------------------------------------===//

struct Conflict {
  std::string Uri;
  int64_t Offset;
  std::string Message;
  std::string Serialized;
  json::Value Result = nullptr;

  bool operator<(const Conflict &Other) const {
    return std::tie(Uri, Offset, Message, Serialized) <
           std::tie(Other.Uri, Other.Offset, Other.Message, Other.Serialized);
  }
};

class ConflictReportMerger {
public:
  void add(json::Value Log, StringRef Path);
  json::Value takeMerged();

private:
  void addConflict(json::Value Result);

  bool HasInput = false;
  json::Object Root;
  json::Object Run;
  std::set<std::string> Seen;
  std::vector<Conflict> Conflicts;
};

void ConflictReportMerger::add(json::Value Log, StringRef Path) {
  json::Object &InputRun = getRun(Log, Path);
  const json::Object *InputTool = InputRun.getObject("tool");
  StringRef InputName =
      getString(InputTool ? InputTool->getObject("driver") : nullptr, "name");

  if (!HasInput) {
    HasInput = true;
    Root = copyExcept(*Log.getAsObject(), {"runs"});
    Run = copyExcept(InputRun, {"results"});
  } else {
    const json::Object *Tool = Run.getObject("tool");
    StringRef Name =
        getString(Tool ? Tool->getObject("driver") : nullptr, "name");
    if (InputName != Name) {
      fail(ErrorMessages::OtherTool, Path, InputName, Name);
    }
  }

  if (json::Array *Results = InputRun.getArray("results")) {
    for (json::Value &Result : *Results) {
      addConflict(std::move(Result));
    }
  }
}

void ConflictReportMerger::addConflict(json::Value Result) {
  Conflict C;
  C.Serialized = formatv("{0}", Result);
  if (!Seen.insert(C.Serialized).second) {
    return;
  }
  C.Offset = 0;
  const json::Object *Object = Result.getAsObject();
  const json::Array *Locations =
      Object ? Object->getArray("locations") : nullptr;
  if (Locations && !Locations->empty()) {
    const json::Object *Location = Locations->front().getAsObject();
    const json::Object *Physical =
        Location ? Location->getObject("physicalLocation") : nullptr;
    if (Physical) {
      C.Uri = getString(Physical->getObject("artifactLocation"), "uri").str();
      if (const json::Object *Region = Physical->getObject("region")) {
        C.Offset = Region->getInteger("byteOffset").value_or(0);
      }
    }
  }
  C.Message =
      getString(Object ? Object->getObject("message") : nullptr, "text").str();
  C.Result = std::move(Result);
  Conflicts.push_back(std::move(C));
}

json::Value ConflictReportMerger::takeMerged() {
  llvm::sort(Conflicts);
  json::Array Results;
  for (Conflict &C : Conflicts) {
    Results.push_back(std::move(C.Result));
  }
  Run["results"] = std::move(Results);
  json::Array Runs;
  Runs.push_back(std::move(Run));
  Root["runs"] = std::move(Runs);
  return std::move(Root);
}

} // namespace

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

int main(int argc, const char **argv) {
  llvm::StringRef ToolHeading = "SSAF Conflict Report Merger";

  InitLLVM X(argc, argv);
  initTool(argc, argv, "0.1", MergeConflictReportCategory, ToolHeading);

  ConflictReportMerger Merger;
  for (const std::string &Path : InputPaths) {
    Merger.add(readLog(Path), Path);
  }
  writeLog(Merger.takeMerged(), OutputPath);

  return 0;
}
