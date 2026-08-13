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
// Conflict policy: this tool implements a drop-all policy. After
// mergeAndDeduplicate runs, the tool computes conflict clusters from the
// input keys — a cluster being a maximal connected component of input
// Replacements (within one file) whose [offset, offset+length) byte ranges
// transitively overlap, restricted to length > 0 entries. For each
// reportable cluster (one whose members intersect the library's
// merged-output keys), every member is removed from the merged YAML,
// one summary line is emitted to stderr, and the tool exits 0.
//
// Zero-length insertions are out of scope for the drop-all policy in this
// change; they continue to follow clang-apply-replacements' own
// IgnoreInsertConflict=false (first-registered) policy.
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
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace cl = llvm::cl;

cl::OptionCategory MergeCategory("clang-ssaf-src-edit-merge options");

cl::list<std::string> InputFiles(cl::Positional, cl::OneOrMore,
                                 cl::desc("<input.yaml>..."),
                                 cl::cat(MergeCategory));

cl::opt<std::string> OutputFile("o", cl::Required, cl::value_desc("path"),
                                cl::desc("Output path for the merged YAML."),
                                cl::cat(MergeCategory));

cl::opt<std::string> SarifConflictsOut(
    "sarif-conflicts-out", cl::value_desc("path"),
    cl::desc("Optional path. When supplied, write a SARIF 2.1.0 document "
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
    llvm::errs() << "clang-ssaf-src-edit-merge: cannot read " << Path << ": "
                 << EC.message() << "\n";
    return false;
  }
  llvm::yaml::Input YAML(Buffer.get()->getBuffer());
  YAML >> Out;
  if (YAML.error()) {
    llvm::errs() << "clang-ssaf-src-edit-merge: " << Path
                 << ": invalid TranslationUnitReplacements YAML\n";
    return false;
  }
  return true;
}

/// Identifying tuple for a replacement, used for both conflict-cluster
/// analysis (input side) and drop-all filtering against the library's
/// merged-output keys.
struct ReplacementKey {
  std::string FilePath;
  unsigned Offset;
  unsigned Length;
  std::string Text;

  bool operator<(const ReplacementKey &Other) const {
    if (FilePath != Other.FilePath)
      return FilePath < Other.FilePath;
    if (Offset != Other.Offset)
      return Offset < Other.Offset;
    if (Length != Other.Length)
      return Length < Other.Length;
    return Text < Other.Text;
  }
};

ReplacementKey makeKey(const clang::tooling::Replacement &R) {
  return ReplacementKey{R.getFilePath().str(), R.getOffset(), R.getLength(),
                        R.getReplacementText().str()};
}

/// Pull every Replacement out of a FileToChangesMap into a flat ordered
/// vector. Within each AtomicChange, the underlying Replacements are
/// already (file, offset)-ordered by the library; across files we order
/// by file path for determinism.
std::vector<clang::tooling::Replacement>
flattenFileChanges(const clang::replace::FileToChangesMap &Changes) {
  // Group by file first so the output is deterministic across runs.
  std::map<std::string, std::vector<clang::tooling::Replacement>> ByFile;
  for (const auto &Entry : Changes) {
    const std::string Path = Entry.first.getName().str();
    auto &Bucket = ByFile[Path];
    for (const clang::tooling::AtomicChange &AC : Entry.second) {
      for (const clang::tooling::Replacement &R : AC.getReplacements())
        Bucket.push_back(R);
    }
  }

  std::vector<clang::tooling::Replacement> Flat;
  for (auto &Entry : ByFile) {
    auto &Bucket = Entry.second;
    // Sort within a file by (offset, length, text) so per-file order is
    // deterministic regardless of input order.
    llvm::sort(Bucket, [](const clang::tooling::Replacement &A,
                          const clang::tooling::Replacement &B) {
      if (A.getOffset() != B.getOffset())
        return A.getOffset() < B.getOffset();
      if (A.getLength() != B.getLength())
        return A.getLength() < B.getLength();
      return A.getReplacementText() < B.getReplacementText();
    });
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
/// A cluster is a maximal connected component of input replacements (within
/// one file) whose [offset, offset+length) byte ranges transitively overlap.
/// Only length > 0 keys participate; zero-length insertions are out of scope
/// for the drop-all policy per spec and continue to follow the library's
/// existing IgnoreInsertConflict=false (first-registered) policy.
///
/// The walk groups by file, sorts each file's keys by (offset, length), and
/// merges into the current cluster whenever
///   key.offset < lastEnd, where
///   lastEnd = max(member.offset + member.length) across cluster members.
/// Otherwise the current cluster closes and a new one opens.
///
/// Only clusters of size > 1 are returned — singletons are not conflicts.
///
/// Each returned cluster's member list is sorted by (offset, length, text);
/// the cluster list itself is sorted by (file, min-offset) ascending. This
/// pins iteration order for both stderr cluster lines and (in a follow-on
/// task) the SARIF results array, satisfying the spec's argv-permutation
/// invariance promise.
std::vector<std::vector<ReplacementKey>>
buildConflictClusters(const std::set<ReplacementKey> &InputKeys) {
  // Group length > 0 keys by file.
  std::map<std::string, std::vector<ReplacementKey>> ByFile;
  for (const ReplacementKey &K : InputKeys) {
    if (K.Length == 0)
      continue;
    ByFile[K.FilePath].push_back(K);
  }

  std::vector<std::vector<ReplacementKey>> Clusters;
  for (auto &Entry : ByFile) {
    auto &Keys = Entry.second;
    // InputKeys is a std::set sorted by (file, offset, length, text), so the
    // per-file vector is already in (offset, length, text) order. Sort
    // defensively for clarity.
    llvm::sort(Keys, [](const ReplacementKey &A, const ReplacementKey &B) {
      if (A.Offset != B.Offset)
        return A.Offset < B.Offset;
      if (A.Length != B.Length)
        return A.Length < B.Length;
      return A.Text < B.Text;
    });

    std::vector<ReplacementKey> Current;
    unsigned LastEnd = 0;
    auto Flush = [&]() {
      if (Current.size() > 1)
        Clusters.push_back(std::move(Current));
      Current.clear();
      LastEnd = 0;
    };

    for (const ReplacementKey &K : Keys) {
      if (Current.empty()) {
        Current.push_back(K);
        LastEnd = K.Offset + K.Length;
        continue;
      }
      if (K.Offset < LastEnd) {
        Current.push_back(K);
        LastEnd = std::max(LastEnd, K.Offset + K.Length);
      } else {
        Flush();
        Current.push_back(K);
        LastEnd = K.Offset + K.Length;
      }
    }
    Flush();
  }

  // Pin cluster-list order by (file, min-offset) ascending.
  llvm::sort(Clusters, [](const std::vector<ReplacementKey> &A,
                          const std::vector<ReplacementKey> &B) {
    if (A.front().FilePath != B.front().FilePath)
      return A.front().FilePath < B.front().FilePath;
    return A.front().Offset < B.front().Offset;
  });

  return Clusters;
}

/// Emit one stderr line per reportable conflict cluster.
///
/// Format (preserved verbatim from the prior implementation):
///   conflict: skipped <count> overlapping replacement(s) at <file>:<offset>
/// where <count> is |cluster| (the full number of dropped members — under
/// drop-all this is the cluster size, NOT cluster size − 1 as it was when
/// first-registered let one survive) and <offset> is the cluster's minimum
/// offset.
///
/// Caller is responsible for passing only reportable clusters
/// (cluster ∩ OutputKeys ≠ ∅) and for the (file, min-offset) sort.
void emitConflictClusterLines(
    const std::vector<std::vector<ReplacementKey>> &Clusters) {
  for (const auto &Cluster : Clusters) {
    llvm::errs() << "conflict: skipped " << Cluster.size()
                 << " overlapping replacement(s) at "
                 << Cluster.front().FilePath << ":" << Cluster.front().Offset
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

/// Emit a SARIF 2.1.0 document at `Path` listing every conflict cluster.
///
/// `Clusters` SHALL be pre-sorted by `(file, min-offset)` ascending by the
/// caller; this emitter walks them in order to populate
/// `runs[0].results[]`. Within each cluster, `relatedLocations[]` is
/// sorted locally by `(byteLength, candidate-text)` ascending per the
/// "SARIF conflict report" requirement — this is a different sort from
/// the cluster's outer `(offset, length, text)` ordering, so members are
/// re-sorted inside this helper.
///
/// Even when `Clusters` is empty, this writes a well-formed SARIF
/// document with `runs[0].results: []`. The file's presence is the
/// "merger ran with conflict reporting requested" signal.
llvm::Error
emitConflictSarif(llvm::StringRef Path,
                  llvm::ArrayRef<std::vector<ReplacementKey>> Clusters) {
  llvm::json::Array Results;
  Results.reserve(Clusters.size());

  for (const auto &Cluster : Clusters) {
    const ReplacementKey &Min = Cluster.front();
    std::string Uri = canonicalizeToFileUri(Min.FilePath);

    // Re-sort cluster members locally by (byteLength, text) ascending for
    // argv-permutation invariance of relatedLocations[].
    std::vector<ReplacementKey> Sorted(Cluster.begin(), Cluster.end());
    llvm::sort(Sorted, [](const ReplacementKey &A, const ReplacementKey &B) {
      if (A.Length != B.Length)
        return A.Length < B.Length;
      return A.Text < B.Text;
    });

    llvm::json::Array RelatedLocations;
    RelatedLocations.reserve(Sorted.size());
    for (size_t I = 0; I < Sorted.size(); ++I) {
      const ReplacementKey &K = Sorted[I];
      RelatedLocations.push_back(llvm::json::Object{
          {"id", static_cast<int64_t>(I + 1)},
          {"physicalLocation",
           llvm::json::Object{
               {"artifactLocation", llvm::json::Object{{"uri", Uri}}},
               {"region",
                llvm::json::Object{
                    {"byteOffset", static_cast<int64_t>(K.Offset)},
                    {"byteLength", static_cast<int64_t>(K.Length)}}}}},
          {"message", llvm::json::Object{
                          {"text", ("candidate edit: \"" + K.Text + "\"")}}}});
    }

    std::string MessageText =
        llvm::formatv("{0} overlapping replacement(s) at {1} byte {2} were "
                      "dropped; resolve manually.",
                      Cluster.size(), Uri, Min.Offset)
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
                  {"region",
                   llvm::json::Object{
                       {"byteOffset", static_cast<int64_t>(Min.Offset)}}}}}}}},
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
                 llvm::json::Object{{"name", "clang-ssaf-src-edit-merge"},
                                    {"version", CLANG_VERSION_STRING}}}}},
           {"results", std::move(Results)}}}}};

  std::error_code EC;
  llvm::raw_fd_ostream OS(Path, EC, llvm::sys::fs::OF_Text);
  if (EC)
    return llvm::createStringError(EC, "cannot write " + Path);
  // Pretty-print with indent 2 via the json::Value format_provider.
  OS << llvm::formatv("{0:2}", Doc) << "\n";
  OS.flush();
  if (OS.has_error())
    return llvm::createStringError(OS.error(), "write error on " + Path);
  return llvm::Error::success();
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

  // Read all inputs.
  std::vector<clang::tooling::TranslationUnitReplacements> TUs;
  TUs.reserve(InputFiles.size());
  for (const std::string &Path : InputFiles) {
    clang::tooling::TranslationUnitReplacements TU;
    if (!readInput(Path, TU))
      return 1;
    TUs.push_back(std::move(TU));
  }

  // Pre-deduplicate identical replacements across all input TUs before the
  // library merge. clang-apply-replacements' groupReplacements only dedups
  // TUDiagnostics-sourced replacements; plain TUReplacements are appended
  // unconditionally, so identical zero-length inserts at the same offset
  // (e.g., a `.data()` rewrite in a header included by N TUs, or an
  // `addr_of(...)` wrap closing-paren) get stacked by AtomicChange::replace
  // into runaway `.data().data()...` or `))))` chains. The first occurrence
  // (in input-file order, then within-file order) wins; later duplicates are
  // byte-identical to it in (file, offset, length, text), so the choice is
  // observationally moot.
  {
    std::set<ReplacementKey> SeenKeys;
    for (auto &TU : TUs) {
      std::vector<clang::tooling::Replacement> Unique;
      Unique.reserve(TU.Replacements.size());
      for (const clang::tooling::Replacement &R : TU.Replacements) {
        if (SeenKeys.insert(makeKey(R)).second)
          Unique.push_back(R);
      }
      TU.Replacements = std::move(Unique);
    }
  }

  // Pre-compute the input-side replacement set for conflict reporting.
  std::set<ReplacementKey> InputKeys;
  for (const auto &TU : TUs)
    for (const auto &R : TU.Replacements)
      InputKeys.insert(makeKey(R));

  // Build a SourceManager for mergeAndDeduplicate.
  clang::DiagnosticOptions DiagOpts;
  clang::DiagnosticsEngine Diagnostics(clang::DiagnosticIDs::create(),
                                       DiagOpts);
  clang::FileManager Files((clang::FileSystemOptions()));
  clang::SourceManager SM(Diagnostics, Files);

  // Run the library's merge. mergeAndDeduplicate's return value (true = no
  // overlap detected, false = at least one overlap dropped) is no longer
  // consulted directly. The tool's drop-all policy operates on its own
  // cluster analysis below; the library's first-registered behavior is
  // overridden by removing every cluster member from OutDoc.Replacements.
  // The library's return value still drives its own per-Replacement stderr
  // diagnostics, which we leave intact.
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
  std::vector<std::vector<ReplacementKey>> Clusters =
      buildConflictClusters(InputKeys);

  // Step 2: compute OutputKeys = set of ReplacementKey for every entry in
  // the library's merged FileChanges, BEFORE drop-all filtering. This is
  // the cluster-eligibility predicate's right-hand side.
  std::set<ReplacementKey> OutputKeys;
  for (const auto &R : OutDoc.Replacements)
    OutputKeys.insert(makeKey(R));

  // Step 3: a cluster is reportable iff at least one of its members appears
  // in OutputKeys (cluster ∩ OutputKeys ≠ ∅). This eligibility predicate
  // replaces the old `!MergeOk` gate: file-not-found inputs are silently
  // filtered by the library during groupReplacements, so neither member
  // appears in OutputKeys, so the cluster drops out of the report path
  // naturally — no need to consult mergeAndDeduplicate's return value.
  std::vector<std::vector<ReplacementKey>> ReportableClusters;
  ReportableClusters.reserve(Clusters.size());
  for (auto &Cluster : Clusters) {
    bool Reportable = false;
    for (const ReplacementKey &K : Cluster) {
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
  // full ReplacementKey (not just (file, offset)) matters because a
  // zero-length insertion can share an offset with an unrelated conflict
  // cluster (zero-length ranges never overlap anything, so they're never
  // cluster members) — keying on (file, offset) alone would collaterally
  // delete that insertion too.
  std::set<ReplacementKey> KeysToRemove;
  for (const auto &Cluster : ReportableClusters)
    for (const ReplacementKey &K : Cluster)
      KeysToRemove.insert(K);

  if (!KeysToRemove.empty()) {
    auto &Reps = OutDoc.Replacements;
    Reps.erase(std::remove_if(Reps.begin(), Reps.end(),
                              [&](const clang::tooling::Replacement &R) {
                                return KeysToRemove.count(makeKey(R)) > 0;
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
      llvm::errs() << "clang-ssaf-src-edit-merge: "
                   << llvm::toString(std::move(E)) << "\n";
      return 1;
    }
  }

  // Write merged YAML (truncate-and-overwrite per spec).
  std::error_code EC;
  llvm::raw_fd_ostream OutStream(OutputFile, EC, llvm::sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << "clang-ssaf-src-edit-merge: cannot write " << OutputFile
                 << ": " << EC.message() << "\n";
    return 1;
  }
  llvm::yaml::Output YAML(OutStream);
  YAML << OutDoc;
  OutStream.flush();
  if (OutStream.has_error()) {
    llvm::errs() << "clang-ssaf-src-edit-merge: write error on " << OutputFile
                 << "\n";
    return 1;
  }

  return 0;
}
