//===- SrcEditMerge.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// clang-ssaf-src-edit-merge: per-LU source-edit YAML merge tool.
//
// Reads N per-TU clang::tooling::TranslationUnitReplacements YAML files,
// deduplicates and merges them via the in-tree clang-apply-replacements
// library (specifically clang::replace::mergeAndDeduplicate), and writes a
// single merged YAML. The tool does NOT write source files — applying the
// merge result is the caller's responsibility (typically clang-reforge
// invokes `clang-apply-replacements` after this tool returns).
//
// Conflict policy: this tool implements a drop-all policy on top of the
// underlying merge step, which resolves overlapping Replacements itself by
// keeping the first-registered one in each overlapping group, and silently
// excludes any input file it could not process at all (e.g. one that does
// not exist on disk).
//
//===----------------------------------------------------------------------===//

#include "clang-apply-replacements/Tooling/ApplyReplacements.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/Tooling/ReplacementsYaml.h" // IWYU pragma: keep
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace cl = llvm::cl;

//===----------------------------------------------------------------------===//
// Error Messages
//===----------------------------------------------------------------------===//

constexpr const char *ToolName = "clang-ssaf-src-edit-merge";

constexpr const char *CannotReadInput = "cannot read {0}: {1}";

constexpr const char *InvalidReplacementsYaml =
    "{0}: invalid TranslationUnitReplacements YAML";

constexpr const char *ConflictClusterSummary =
    "conflict: skipped {0} overlapping replacement(s) at {1}:{2}";

constexpr const char *CannotWriteFile = "cannot write {0}";

constexpr const char *WriteErrorOnFile = "write error on {0}";

constexpr const char *CannotWriteOutput = "cannot write {0}: {1}";

constexpr const char *CandidateEditMessage = "candidate edit: \"{0}\"";

constexpr const char *ConflictSarifMessage =
    "{0} overlapping replacement(s) at {1} byte {2} were dropped; resolve "
    "manually.";

cl::OptionCategory MergeCategory("clang-ssaf-src-edit-merge options");

cl::list<std::string> InputFiles(cl::Positional, cl::OneOrMore,
                                 cl::desc("<input.yaml>..."),
                                 cl::cat(MergeCategory));

cl::opt<std::string> OutputFile("o", cl::Required, cl::value_desc("path"),
                                cl::desc("Output path for the merged YAML."),
                                cl::cat(MergeCategory));

cl::opt<std::string> SarifConflictsOut(
    "sarif-conflicts-out", cl::value_desc("path"),
    cl::desc("Optional path. When supplied, write a SARIF document "
             "listing conflict clusters dropped from the merged output."),
    cl::cat(MergeCategory));

/// Read one input YAML into a TranslationUnitReplacements.
///
/// Returns true on success. On failure, prints a one-line diagnostic to
/// stderr and returns false; the caller surfaces this as a non-zero exit.
bool readInput(llvm::StringRef Path,
               clang::tooling::TranslationUnitReplacements &Out) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
      llvm::MemoryBuffer::getFile(Path);
  if (std::error_code EC = Buffer.getError()) {
    llvm::errs() << ToolName << ": "
                 << llvm::formatv(CannotReadInput, Path, EC.message()) << "\n";
    return false;
  }
  llvm::yaml::Input YAML(Buffer.get()->getBuffer());
  YAML >> Out;
  if (YAML.error()) {
    llvm::errs() << ToolName << ": "
                 << llvm::formatv(InvalidReplacementsYaml, Path) << "\n";
    return false;
  }
  return true;
}

/// Pull every Replacement out of a FileToChangesMap into a flat ordered
/// vector. Within each AtomicChange, the underlying Replacements are
/// already (file, offset)-ordered by the library; across files we order
/// by file path for determinism.
std::vector<clang::tooling::Replacement>
flattenFileChanges(const clang::replace::FileToChangesMap &Changes) {
  // Group by file first so the output is deterministic across runs.
  std::map<std::string, std::set<clang::tooling::Replacement>> ByFile;
  for (const auto &Entry : Changes) {
    const std::string Path = Entry.first.getName().str();
    auto &Bucket = ByFile[Path];
    for (const clang::tooling::AtomicChange &AC : Entry.second) {
      for (const clang::tooling::Replacement &R : AC.getReplacements())
        Bucket.insert(R);
    }
  }

  std::vector<clang::tooling::Replacement> Flat;
  for (auto &Entry : ByFile) {
    // Bucket is a std::set<Replacement>, so it's already ordered by
    // Replacement::operator< — the (offset, length, text) order this
    // function promises, since every entry here shares Entry.first as its
    // file path.
    auto &Bucket = Entry.second;
    for (const auto &R : Bucket)
      Flat.push_back(R);
  }
  return Flat;
}

/// Compute the shared MainSourceFile across inputs.
///
/// Per spec: if every input declares the same MainSourceFile, use that;
/// otherwise use the empty string.
std::string computeMainSourceFile(
    const std::vector<clang::tooling::TranslationUnitReplacements> &TUs) {
  if (TUs.empty())
    return "";
  const std::string &First = TUs.front().MainSourceFile;
  for (const auto &TU : TUs)
    if (TU.MainSourceFile != First)
      return "";
  return First;
}

/// Build the conflict cluster list from the merged input key set.
///
/// `InputKeysByFile` is every input Replacement, grouped by file — the
/// caller does this grouping once, up front, rather than this function
/// flattening across files just to immediately re-group by file itself.
///
/// A cluster is a maximal connected component of one file's input
/// replacements whose [offset, offset+length) byte ranges transitively
/// overlap. Only length > 0 keys participate; zero-length insertions are
/// out of scope for the drop-all policy per spec and continue to follow the
/// library's existing IgnoreInsertConflict=false (first-registered) policy.
///
/// The walk merges into the current cluster whenever
///   key.offset < lastEnd, where
///   lastEnd = max(member.offset + member.length) across cluster members.
/// Otherwise the current cluster closes and a new one opens.
///
/// Only clusters of size > 1 are returned; singletons are not conflicts.
///
/// Each returned cluster's member list is sorted by (offset, length, text);
/// the cluster list itself is sorted by (file, min-offset) ascending. This
/// pins iteration order for both stderr cluster lines and (in a follow-on
/// task) the SARIF results array.
std::vector<std::vector<clang::tooling::Replacement>> buildConflictClusters(
    const std::map<std::string, std::set<clang::tooling::Replacement>>
        &InputKeysByFile) {
  std::vector<std::vector<clang::tooling::Replacement>> Clusters;
  for (auto &Entry : InputKeysByFile) {
    // Keys is a std::set<Replacement>, so it's already ordered by
    // Replacement::operator< — the (offset, length, text) order the cluster
    // walk below needs, since every entry here shares Entry.first as its
    // file path.
    auto &Keys = Entry.second;

    std::vector<clang::tooling::Replacement> Current;
    unsigned LastEnd = 0;
    auto Flush = [&]() {
      if (Current.size() > 1)
        Clusters.push_back(std::move(Current));
      Current.clear();
      LastEnd = 0;
    };

    for (const clang::tooling::Replacement &K : Keys) {
      if (K.getLength() == 0)
        continue;
      if (Current.empty()) {
        Current.push_back(K);
        LastEnd = K.getOffset() + K.getLength();
        continue;
      }
      if (K.getOffset() < LastEnd) {
        Current.push_back(K);
        LastEnd = std::max(LastEnd, K.getOffset() + K.getLength());
      } else {
        Flush();
        Current.push_back(K);
        LastEnd = K.getOffset() + K.getLength();
      }
    }
    Flush();
  }

  // Pin cluster-list order by (file, min-offset) ascending.
  llvm::sort(Clusters, [](const std::vector<clang::tooling::Replacement> &A,
                          const std::vector<clang::tooling::Replacement> &B) {
    if (A.front().getFilePath() != B.front().getFilePath())
      return A.front().getFilePath() < B.front().getFilePath();
    return A.front().getOffset() < B.front().getOffset();
  });

  return Clusters;
}

/// Emit one stderr line per reportable conflict cluster.
///
/// Precondition: `Clusters` contains only reportable clusters (those with a
/// member in OutputKeys) and is sorted by (file, min-offset)
void emitConflictClusterLines(
    const std::vector<std::vector<clang::tooling::Replacement>> &Clusters) {
  for (const auto &Cluster : Clusters) {
    llvm::errs() << llvm::formatv(ConflictClusterSummary, Cluster.size(),
                                  Cluster.front().getFilePath(),
                                  Cluster.front().getOffset())
                 << "\n";
  }
}

/// Canonicalize a Replacement's `FilePath` into an absolute `file://` URI.
///
/// Fallback chain:
///   1. `llvm::sys::fs::real_path` — resolves symlinks and yields an
///      absolute path. Only succeeds if the file exists on disk.
///   2. `llvm::sys::fs::make_absolute` — succeeds for non-existent paths
///      too; used for synthetic test fixtures whose FilePath may name a
///      file that the merger never opened.
///   3. Raw `FilePath` — last-resort fallback if both of the above fail.
///      Emits a syntactically valid `file://` URI even if the underlying
///      path is relative, matching the SARIF requirement's "absolute"
///      promise loosely (downstream tooling that needs strict absolute
///      URIs SHOULD canonicalize on its end if the disk state permits).
std::string canonicalizeToFileUri(llvm::StringRef FilePath) {
  llvm::SmallString<256> Buf;
  if (!llvm::sys::fs::real_path(FilePath, Buf))
    return "file://" + llvm::sys::path::convert_to_slash(Buf);
  Buf.assign(FilePath.begin(), FilePath.end());
  if (!llvm::sys::fs::make_absolute(Buf))
    return "file://" + llvm::sys::path::convert_to_slash(Buf);
  return "file://" + llvm::sys::path::convert_to_slash(FilePath);
}

/// Emit a SARIF document at `Path` listing every conflict cluster.
///
/// `Clusters` SHALL be pre-sorted by `(file, min-offset)` ascending by the
/// caller; this emitter walks them in order to populate
/// `runs[0].results[]`. Within each cluster, `relatedLocations[]` is
/// sorted locally by `(byteLength, candidate-text)` ascending per the
/// "SARIF conflict report" requirement.
///
/// Even when `Clusters` is empty, this writes a well-formed SARIF
/// document with `runs[0].results: []`. The file's presence is the
/// "merger ran with conflict reporting requested" signal.
llvm::Error emitConflictSarif(
    llvm::StringRef Path,
    llvm::ArrayRef<std::vector<clang::tooling::Replacement>> Clusters) {
  llvm::json::Array Results;
  Results.reserve(Clusters.size());

  for (const auto &Cluster : Clusters) {
    const clang::tooling::Replacement &Min = Cluster.front();
    std::string Uri = canonicalizeToFileUri(Min.getFilePath());

    // Re-sort cluster members locally by (byteLength, text) ascending.
    std::vector<clang::tooling::Replacement> Sorted(Cluster.begin(),
                                                    Cluster.end());
    llvm::sort(Sorted, [](const clang::tooling::Replacement &A,
                          const clang::tooling::Replacement &B) {
      if (A.getLength() != B.getLength())
        return A.getLength() < B.getLength();
      return A.getReplacementText() < B.getReplacementText();
    });

    llvm::json::Array RelatedLocations;
    RelatedLocations.reserve(Sorted.size());
    for (size_t I = 0; I < Sorted.size(); ++I) {
      const clang::tooling::Replacement &K = Sorted[I];
      RelatedLocations.push_back(llvm::json::Object{
          {"id", static_cast<int64_t>(I + 1)},
          {"physicalLocation",
           llvm::json::Object{
               {"artifactLocation", llvm::json::Object{{"uri", Uri}}},
               {"region",
                llvm::json::Object{
                    {"byteOffset", static_cast<int64_t>(K.getOffset())},
                    {"byteLength", static_cast<int64_t>(K.getLength())}}}}},
          {"message",
           llvm::json::Object{{"text", llvm::formatv(CandidateEditMessage,
                                                     K.getReplacementText())
                                           .str()}}}});
    }

    std::string MessageText =
        llvm::formatv(ConflictSarifMessage, Cluster.size(), Uri,
                      Min.getOffset())
            .str();

    Results.push_back(llvm::json::Object{
        {"ruleId", "clang-reforge-replacement-conflict"},
        {"level", "error"},
        {"message", llvm::json::Object{{"text", MessageText}}},
        {"locations",
         llvm::json::Array{llvm::json::Object{
             {"physicalLocation",
              llvm::json::Object{
                  {"artifactLocation", llvm::json::Object{{"uri", Uri}}},
                  {"region", llvm::json::Object{{"byteOffset",
                                                 static_cast<int64_t>(
                                                     Min.getOffset())}}}}}}}},
        {"relatedLocations", std::move(RelatedLocations)}});
  }

  llvm::json::Value Doc = llvm::json::Object{
      {"version", "2.1.0"},
      {"$schema", "https://json.schemastore.org/sarif-2.1.0.json"},
      {"runs",
       llvm::json::Array{llvm::json::Object{
           {"tool",
            llvm::json::Object{
                {"driver",
                 llvm::json::Object{{"name", ToolName},
                                    {"version", CLANG_VERSION_STRING}}}}},
           {"results", std::move(Results)}}}}};

  std::error_code EC;
  llvm::raw_fd_ostream OS(Path, EC, llvm::sys::fs::OF_Text);
  if (EC)
    return llvm::createStringError(EC,
                                   llvm::formatv(CannotWriteFile, Path).str());
  // Pretty-print with indent 2 via the json::Value format_provider.
  OS << llvm::formatv("{0:2}", Doc) << "\n";
  OS.flush();
  if (OS.has_error())
    return llvm::createStringError(OS.error(),
                                   llvm::formatv(WriteErrorOnFile, Path).str());
  return llvm::Error::success();
}

/// Returns whether `Path`'s parent directory exists, so a bad `-o` or
/// `--sarif-conflicts-out` path can be rejected before any merge work runs.
/// A `Path` with no directory component (e.g. a bare file name) is treated
/// as valid — it names a file in the current directory. This is a
/// best-effort check, not a substitute for handling the real open() failure:
/// it cannot catch permission errors or a race between the check and the
/// eventual write.
bool parentDirectoryExists(llvm::StringRef Path) {
  llvm::StringRef Parent = llvm::sys::path::parent_path(Path);
  return Parent.empty() || llvm::sys::fs::is_directory(Parent);
}

} // namespace

int main(int argc, const char **argv) {
  llvm::InitLLVM X(argc, argv);
  cl::HideUnrelatedOptions(MergeCategory);
  cl::ParseCommandLineOptions(
      argc, argv,
      "clang-ssaf-src-edit-merge: merge per-TU TranslationUnitReplacements "
      "YAML files for one link unit into a single merged YAML. Does not "
      "write source files; the apply step is the caller's responsibility.\n");

  // Validate the command-line parameters that can be checked without
  // reading any input, so a bad -o or --sarif-conflicts-out path is rejected
  // before the (potentially expensive) merge work below runs.
  if (!parentDirectoryExists(OutputFile)) {
    llvm::errs() << ToolName << ": "
                 << llvm::formatv(CannotWriteFile, OutputFile) << "\n";
    return 1;
  }
  if (!SarifConflictsOut.empty() && !parentDirectoryExists(SarifConflictsOut)) {
    llvm::errs() << ToolName << ": "
                 << llvm::formatv(CannotWriteFile, SarifConflictsOut) << "\n";
    return 1;
  }

  // Read all inputs.
  std::vector<clang::tooling::TranslationUnitReplacements> TUs;
  TUs.reserve(InputFiles.size());
  for (const std::string &Path : InputFiles) {
    clang::tooling::TranslationUnitReplacements TU;
    if (!readInput(Path, TU))
      return 1;
    TUs.push_back(std::move(TU));
  }

  // Pre-deduplicate identical replacements across all input TUs before
  // calling into clang-apply-replacements.
  //
  // This loop keeps a running set of every (file, offset, length, text)
  // tuple already kept across all TUs and drops
  // any later Replacement that matches one already kept, so each distinct
  // Replacement reaches the library exactly once. The first occurrence (in
  // input-file order, then within-file order) wins; later duplicates are
  // byte-identical to it, so which one is "first" is observationally moot.
  {
    std::set<clang::tooling::Replacement> SeenKeys;
    for (auto &TU : TUs) {
      std::vector<clang::tooling::Replacement> Unique;
      Unique.reserve(TU.Replacements.size());
      for (const clang::tooling::Replacement &R : TU.Replacements) {
        if (SeenKeys.insert(R).second)
          Unique.push_back(R);
      }
      TU.Replacements = std::move(Unique);
    }
  }

  // Pre-compute the input-side replacement set for conflict reporting,
  // grouped by file up front since buildConflictClusters only looks for
  // overlap within one file.
  std::map<std::string, std::set<clang::tooling::Replacement>> InputKeysByFile;
  for (const auto &TU : TUs)
    for (const auto &R : TU.Replacements)
      InputKeysByFile[R.getFilePath().str()].insert(R);

  // Build a SourceManager for mergeAndDeduplicate.
  clang::DiagnosticOptions DiagOpts;
  clang::DiagnosticsEngine Diagnostics(clang::DiagnosticIDs::create(),
                                       DiagOpts);
  clang::FileManager Files((clang::FileSystemOptions()));
  clang::SourceManager SM(Diagnostics, Files);

  // Run the library's merge. The tool's drop-all policy operates on its own
  // cluster analysis below; the mergeAndDeduplicate library's first-registered
  // behavior is overridden by removing every cluster member from
  // OutDoc.Replacements. The library's return value still drives its own
  // per-Replacement stderr diagnostics, which we leave intact.
  clang::replace::FileToChangesMap FileChanges;
  const clang::replace::TUDiagnostics NoDiagnostics;
  (void)clang::replace::mergeAndDeduplicate(TUs, NoDiagnostics, FileChanges, SM,
                                            /*IgnoreInsertConflict=*/false);

  // Flatten the merged FileChanges back into a TranslationUnitReplacements.
  clang::tooling::TranslationUnitReplacements OutDoc;
  OutDoc.MainSourceFile = computeMainSourceFile(TUs);
  OutDoc.Replacements = flattenFileChanges(FileChanges);

  // Drop-all conflict handling.
  //
  // Step 1: build conflict clusters from the input key set.
  std::vector<std::vector<clang::tooling::Replacement>> Clusters =
      buildConflictClusters(InputKeysByFile);

  // Step 2: compute OutputKeys = set of Replacement for every entry in
  // the library's merged FileChanges, BEFORE drop-all filtering. This is
  // the cluster-eligibility predicate's right-hand side.
  std::set<clang::tooling::Replacement> OutputKeys;
  for (const auto &R : OutDoc.Replacements)
    OutputKeys.insert(R);

  // Step 3: a cluster is reportable iff at least one of its members appears
  // in OutputKeys (cluster ∩ OutputKeys ≠ ∅).
  std::vector<std::vector<clang::tooling::Replacement>> ReportableClusters;
  ReportableClusters.reserve(Clusters.size());
  for (auto &Cluster : Clusters) {
    bool Reportable = false;
    for (const clang::tooling::Replacement &K : Cluster) {
      if (OutputKeys.count(K)) {
        Reportable = true;
        break;
      }
    }
    if (Reportable)
      ReportableClusters.push_back(std::move(Cluster));
  }

  // Step 4: build the exact key set for every reportable cluster member.
  // Drop-all then strips every matching entry from OutDoc. Keying on the
  // full Replacement (not just (file, offset)) matters because a
  // zero-length insertion can share an offset with an unrelated conflict
  // cluster (zero-length ranges never overlap anything, so they're never
  // cluster members) — keying on (file, offset) alone would collaterally
  // delete that insertion too.
  std::set<clang::tooling::Replacement> KeysToRemove;
  for (const auto &Cluster : ReportableClusters)
    for (const clang::tooling::Replacement &K : Cluster)
      KeysToRemove.insert(K);

  if (!KeysToRemove.empty()) {
    auto &Reps = OutDoc.Replacements;
    Reps.erase(std::remove_if(Reps.begin(), Reps.end(),
                              [&](const clang::tooling::Replacement &R) {
                                return KeysToRemove.count(R) > 0;
                              }),
               Reps.end());
  }

  // Step 5: emit stderr cluster lines. ReportableClusters was sorted by
  // (file, min-offset) ascending inside buildConflictClusters; the
  // reportability filter preserved that order.
  emitConflictClusterLines(ReportableClusters);

  // Step 6: when --sarif-conflicts-out=<path> was supplied, write the
  // SARIF document. Empty ReportableClusters still produces a well-formed
  // SARIF with results: [] — the file's presence is the signal that
  // conflict reporting was requested. Flag-omitted skips emission
  // entirely; no file is created at any path.
  if (!SarifConflictsOut.empty()) {
    if (llvm::Error E =
            emitConflictSarif(SarifConflictsOut, ReportableClusters)) {
      llvm::errs() << ToolName << ": " << llvm::toString(std::move(E)) << "\n";
      return 1;
    }
  }

  // Write merged YAML (truncate-and-overwrite per spec).
  std::error_code EC;
  llvm::raw_fd_ostream OutStream(OutputFile, EC, llvm::sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << ToolName << ": "
                 << llvm::formatv(CannotWriteOutput, OutputFile, EC.message())
                 << "\n";
    return 1;
  }
  llvm::yaml::Output YAML(OutStream);
  YAML << OutDoc;
  OutStream.flush();
  if (OutStream.has_error()) {
    llvm::errs() << ToolName << ": "
                 << llvm::formatv(WriteErrorOnFile, OutputFile) << "\n";
    return 1;
  }

  return 0;
}
