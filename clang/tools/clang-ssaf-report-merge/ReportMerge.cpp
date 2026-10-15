//===- ReportMerge.cpp - SSAF Report Merge Tool ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements the tool that merges SARIF transformation reports, as
//  written by clang's --ssaf-transformation-report-file or by this tool, into
//  one report. Every input run numbers its rules and artifacts on its own, so
//  results are renumbered against the merged run, and identical results, such
//  as a finding in a header that several translation units include, are kept
//  once. An artifact is numbered by its own location index where it has one,
//  since clang lists artifacts in URI order but numbers them in the order they
//  were first referred to.
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
#include <map>
#include <optional>
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

cl::OptionCategory ReportMergeCategory("clang-ssaf-report-merge options");

cl::list<std::string> InputPaths(cl::Positional, cl::desc("<input.sarif>..."),
                                 cl::OneOrMore, cl::cat(ReportMergeCategory));

cl::opt<std::string> OutputPath("o", cl::desc("Output file path"),
                                cl::value_desc("path"), cl::Required,
                                cl::cat(ReportMergeCategory));

//===----------------------------------------------------------------------===//
// Error Messages
//===----------------------------------------------------------------------===//

namespace ErrorMessages {

constexpr const char *CannotRead = "failed to read '{0}': {1}";

constexpr const char *CannotParse = "failed to parse '{0}': {1}";

constexpr const char *NotOneRun =
    "'{0}' is not a SARIF log with exactly one run";

constexpr const char *OtherTransformation =
    "'{0}' reports on '{1}', not on '{2}'";

constexpr const char *UnknownRule =
    "'{0}': a result refers to rule {1}, which its run does not define";

constexpr const char *UnknownArtifact =
    "'{0}': a location refers to artifact {1}, which its run does not define";

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

json::Object *getArtifactLocation(json::Value &Location) {
  json::Object *Object = Location.getAsObject();
  json::Object *Physical =
      Object ? Object->getObject("physicalLocation") : nullptr;
  return Physical ? Physical->getObject("artifactLocation") : nullptr;
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

/// A result whose rule and artifacts are named by id and URI rather than by
/// index, so that results from different runs compare equal.
struct Finding {
  std::string Uri;
  int64_t Line;
  int64_t Column;
  std::string RuleId;
  std::string Message;
  std::string Serialized;
  json::Object Result;

  bool operator<(const Finding &Other) const {
    return std::tie(Uri, Line, Column, RuleId, Message, Serialized) <
           std::tie(Other.Uri, Other.Line, Other.Column, Other.RuleId,
                    Other.Message, Other.Serialized);
  }
};

class ReportMerger {
public:
  void add(json::Value Log, StringRef Path);
  json::Value takeMerged();

private:
  void addFinding(json::Object Result);

  bool HasInput = false;
  json::Object Root;
  json::Object Run;
  json::Object Tool;
  json::Object Driver;
  std::map<std::string, json::Value> Rules;
  std::map<std::string, json::Value> Artifacts;
  std::set<std::string> Seen;
  std::vector<Finding> Findings;
};

void ReportMerger::add(json::Value Log, StringRef Path) {
  json::Object &InputRun = getRun(Log, Path);
  const json::Object *InputTool = InputRun.getObject("tool");
  const json::Object *InputDriver =
      InputTool ? InputTool->getObject("driver") : nullptr;

  if (!HasInput) {
    HasInput = true;
    Root = copyExcept(*Log.getAsObject(), {"runs"});
    Run = copyExcept(InputRun, {"artifacts", "results", "tool"});
    if (InputTool) {
      Tool = copyExcept(*InputTool, {"driver"});
    }
    if (InputDriver) {
      Driver = copyExcept(*InputDriver, {"rules"});
    }
  } else if (getString(InputDriver, "name") != getString(&Driver, "name") ||
             getString(InputDriver, "fullName") !=
                 getString(&Driver, "fullName")) {
    fail(ErrorMessages::OtherTransformation, Path,
         getString(InputDriver, "fullName"), getString(&Driver, "fullName"));
  }

  std::vector<std::string> RuleIds;
  if (const json::Array *InputRules =
          InputDriver ? InputDriver->getArray("rules") : nullptr) {
    for (const json::Value &Rule : *InputRules) {
      std::string Id = getString(Rule.getAsObject(), "id").str();
      Rules.try_emplace(Id, Rule);
      RuleIds.push_back(std::move(Id));
    }
  }

  std::map<int64_t, std::string> Uris;
  if (const json::Array *InputArtifacts = InputRun.getArray("artifacts")) {
    for (auto [Position, Artifact] : enumerate(*InputArtifacts)) {
      const json::Object *Object = Artifact.getAsObject();
      const json::Object *Location =
          Object ? Object->getObject("location") : nullptr;
      std::string Uri = getString(Location, "uri").str();
      int64_t Index = static_cast<int64_t>(Position);
      if (Location) {
        Index = Location->getInteger("index").value_or(Index);
      }
      Artifacts.try_emplace(Uri, Artifact);
      Uris[Index] = std::move(Uri);
    }
  }

  json::Array *Results = InputRun.getArray("results");
  if (!Results) {
    return;
  }
  for (json::Value &Value : *Results) {
    json::Object *Result = Value.getAsObject();
    if (!Result) {
      continue;
    }
    if (std::optional<int64_t> Index = Result->getInteger("ruleIndex")) {
      if (*Index < 0 || *Index >= static_cast<int64_t>(RuleIds.size())) {
        fail(ErrorMessages::UnknownRule, Path, *Index);
      }
      Result->try_emplace("ruleId", RuleIds[*Index]);
      Result->erase("ruleIndex");
    }
    for (StringRef Key : {"locations", "relatedLocations"}) {
      json::Array *Locations = Result->getArray(Key);
      if (!Locations) {
        continue;
      }
      for (json::Value &Location : *Locations) {
        json::Object *Artifact = getArtifactLocation(Location);
        if (!Artifact) {
          continue;
        }
        if (std::optional<int64_t> Index = Artifact->getInteger("index")) {
          auto Uri = Uris.find(*Index);
          if (Uri == Uris.end()) {
            fail(ErrorMessages::UnknownArtifact, Path, *Index);
          }
          Artifact->try_emplace("uri", Uri->second);
          Artifact->erase("index");
        }
      }
    }
    addFinding(std::move(*Result));
  }
}

void ReportMerger::addFinding(json::Object Result) {
  json::Value Value(std::move(Result));
  Finding F;
  F.Serialized = formatv("{0}", Value);
  if (!Seen.insert(F.Serialized).second) {
    return;
  }
  json::Object &Object = *Value.getAsObject();
  F.Line = 0;
  F.Column = 0;
  if (json::Array *Locations = Object.getArray("locations");
      Locations && !Locations->empty()) {
    F.Uri = getString(getArtifactLocation(Locations->front()), "uri").str();
    const json::Object *Location = Locations->front().getAsObject();
    const json::Object *Physical =
        Location ? Location->getObject("physicalLocation") : nullptr;
    if (const json::Object *Region =
            Physical ? Physical->getObject("region") : nullptr) {
      F.Line = Region->getInteger("startLine").value_or(0);
      F.Column = Region->getInteger("startColumn").value_or(0);
    }
  }
  F.RuleId = getString(&Object, "ruleId").str();
  F.Message = getString(Object.getObject("message"), "text").str();
  F.Result = std::move(Object);
  Findings.push_back(std::move(F));
}

json::Value ReportMerger::takeMerged() {
  std::map<std::string, int64_t> RuleIndex;
  json::Array MergedRules;
  for (auto &[Id, Rule] : Rules) {
    RuleIndex[Id] = MergedRules.size();
    MergedRules.push_back(std::move(Rule));
  }

  std::map<std::string, int64_t> ArtifactIndex;
  json::Array MergedArtifacts;
  for (auto &[Uri, Artifact] : Artifacts) {
    int64_t Index = MergedArtifacts.size();
    ArtifactIndex[Uri] = Index;
    if (json::Object *Object = Artifact.getAsObject()) {
      if (json::Object *Location = Object->getObject("location")) {
        (*Location)["index"] = Index;
      }
    }
    MergedArtifacts.push_back(std::move(Artifact));
  }

  llvm::sort(Findings);
  json::Array MergedResults;
  for (Finding &F : Findings) {
    auto Rule = RuleIndex.find(F.RuleId);
    if (Rule != RuleIndex.end()) {
      F.Result["ruleIndex"] = Rule->second;
    }
    for (StringRef Key : {"locations", "relatedLocations"}) {
      json::Array *Locations = F.Result.getArray(Key);
      if (!Locations) {
        continue;
      }
      for (json::Value &Location : *Locations) {
        json::Object *Artifact = getArtifactLocation(Location);
        if (!Artifact) {
          continue;
        }
        auto Index = ArtifactIndex.find(getString(Artifact, "uri").str());
        if (Index != ArtifactIndex.end()) {
          (*Artifact)["index"] = Index->second;
        }
      }
    }
    MergedResults.push_back(std::move(F.Result));
  }

  Driver["rules"] = std::move(MergedRules);
  Tool["driver"] = std::move(Driver);
  Run["tool"] = std::move(Tool);
  Run["artifacts"] = std::move(MergedArtifacts);
  Run["results"] = std::move(MergedResults);
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
  llvm::StringRef ToolHeading = "SSAF Transformation Report Merger";

  InitLLVM X(argc, argv);
  initTool(argc, argv, "0.1", ReportMergeCategory, ToolHeading);

  ReportMerger Merger;
  for (const std::string &Path : InputPaths) {
    Merger.add(readLog(Path), Path);
  }
  writeLog(Merger.takeMerged(), OutputPath);

  return 0;
}
