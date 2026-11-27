//===--- RemapMain.cpp - Remap paths in background index shards -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// clangd-remap is a standalone tool that rewrites paths inside every .idx shard
// in a background index directory. An index generated on one machine (or at one
// workspace path) can be remapped and reused within a source tree at a
// different location.
//
// Usage:
//   clangd-remap --path-mappings=/old/root=/new/root /path/to/index-dir
//
//===----------------------------------------------------------------------===//

#include "Headers.h"
#include "PathMapping.h"
#include "SourceCode.h"
#include "URI.h"
#include "index/Ref.h"
#include "index/Serialization.h"
#include "index/Symbol.h"
#include "support/Logger.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <atomic>
#include <string>
#include <vector>

namespace clang {
namespace clangd {
namespace {

static llvm::cl::OptionCategory RemapCategory("clangd-remap options");

static llvm::cl::opt<std::string> PathMappingsArg{
    "path-mappings",
    llvm::cl::cat(RemapCategory),
    llvm::cl::desc(
        "List of path mappings applied to every string in each background "
        "index shard. Format: /old/path=/new/path[,/old2=/new2,...]"),
    llvm::cl::Required,
};

static llvm::cl::opt<std::string> IndexDir{
    llvm::cl::desc("<index-dir>"),
    llvm::cl::cat(RemapCategory),
    llvm::cl::Positional,
    llvm::cl::Required,
};

static llvm::cl::opt<unsigned> NumThreads{
    "j",
    llvm::cl::cat(RemapCategory),
    llvm::cl::desc("Number of worker threads (0 = all)"),
    llvm::cl::init(0),
};

static llvm::cl::opt<Logger::Level> LogLevel{
    "log",
    llvm::cl::cat(RemapCategory),
    llvm::cl::desc("Verbosity of log messages written to stderr"),
    llvm::cl::values(
        clEnumValN(Logger::Error, "error", "Error messages only"),
        clEnumValN(Logger::Info, "info", "High level execution tracing"),
        clEnumValN(Logger::Debug, "verbose", "Low level details")),
    llvm::cl::init(Logger::Info),
};

// Apply a path mapping to a URI or raw path string
//
// Ex. given "-I/old/root/include" and mapping /old/root=/new/root, the result
// is "-I/new/root/include"
std::optional<std::string> remapString(llvm::StringRef S,
                                       const PathMappings &Mappings) {
  // Client = old path, Server = new path; ClientToServer maps old -> new
  if (S.starts_with("file://"))
    return doPathMapping(S, PathMapping::Direction::ClientToServer, Mappings);

  // For non-URI strings (compilation flags, directory paths, etc.) only match
  // at the first '/' (where an absolute path begins)
  // FIXME: This does not handle Windows paths; only POSIX paths are supported.
  size_t FirstSlash = S.find('/');
  if (FirstSlash == llvm::StringRef::npos)
    return std::nullopt;

  for (const auto &Mapping : Mappings) {
    size_t Pos = S.find(Mapping.ClientPath);
    if (Pos == FirstSlash) {
      llvm::StringRef After = S.substr(Pos + Mapping.ClientPath.size());
      // Ensure a full path-component match: "/old" must not match "/older"
      if (After.empty() || After.front() == '/')
        return (S.substr(0, Pos) + Mapping.ServerPath + After).str();
    }
  }
  return std::nullopt;
}

// Remap a StringRef in-place, saving the result into the Arena so the
// pointer remains valid
void remapRef(llvm::StringRef &S, const PathMappings &Mappings,
              llvm::StringSaver &Saver) {
  if (auto R = remapString(S, Mappings))
    S = Saver.save(std::move(*R));
}

// Like remapRef, but _always_ saves into Saver (even on no match). Used for
// StringRefs that will outlive their original storage.
void remapOrCopyRef(llvm::StringRef &S, const PathMappings &Mappings,
                    llvm::StringSaver &Saver) {
  if (auto R = remapString(S, Mappings))
    S = Saver.save(std::move(*R));
  else
    S = Saver.save(S);
}

void remapCharURI(const char *&P, const PathMappings &Mappings,
                  llvm::StringSaver &Saver) {
  llvm::StringRef S(P);
  if (auto R = remapString(S, Mappings))
    P = Saver.save(std::move(*R)).data();
}

void remapStdStr(std::string &S, const PathMappings &Mappings) {
  if (auto R = remapString(S, Mappings))
    S = std::move(*R);
}

std::vector<std::string> collectShards(llvm::StringRef Dir) {
  std::vector<std::string> Paths;
  std::error_code EC;
  for (llvm::sys::fs::recursive_directory_iterator It(Dir, EC), End;
       It != End && !EC; It.increment(EC)) {
    if (llvm::sys::path::extension(It->path()) == ".idx")
      Paths.push_back(It->path());
  }
  if (EC)
    elog("Error scanning directory {0}: {1}", Dir, EC.message());
  return Paths;
}

// Compute shard filename for a source path. (See getShardPathFromFilePath()
// in BackgroundIndexStorage.cpp.)
std::string shardName(llvm::StringRef SourceFilePath) {
  return (llvm::sys::path::filename(SourceFilePath) + "." +
          llvm::toHex(digest(SourceFilePath)) + ".idx")
      .str();
}

// For each source entry, resolve its URI to get the original absolute path and
// compute that shard name. Find the entry whose shard name matches, and apply
// the path mappings to that path to compute the new shard name.
//
// This must be called before remapIndexData(), since it needs the original (not
// remapped) URIs.
std::string deriveNewFilename(const IndexFileIn &Data,
                              llvm::StringRef OldFilename,
                              const PathMappings &Mappings) {
  if (!Data.Sources || Data.Sources->empty())
    return OldFilename.str();

  for (const auto &Entry : *Data.Sources) {
    auto U = URI::parse(Entry.first());
    if (!U) {
      llvm::consumeError(U.takeError());
      continue;
    }
    auto Path = URI::resolve(*U);
    if (!Path) {
      llvm::consumeError(Path.takeError());
      continue;
    }
    if (shardName(*Path) == OldFilename) {
      std::string NewPath = *Path;
      remapStdStr(NewPath, Mappings);
      return shardName(NewPath);
    }
  }
  return OldFilename.str();
}

// Remap all paths inside a parsed IndexFileIn in-place. Saver is used to
// allocate new strings for fields stored as StringRef or raw pointers.
void remapIndexData(IndexFileIn &Data, const PathMappings &Mappings,
                    llvm::StringSaver &Saver) {
  if (Data.Symbols) {
    // SymbolSlab is immutable, so we rebuild it
    SymbolSlab::Builder Builder;
    for (const auto &Sym : *Data.Symbols) {
      Symbol S = Sym;
      remapCharURI(S.CanonicalDeclaration.FileURI, Mappings, Saver);
      remapCharURI(S.Definition.FileURI, Mappings, Saver);
      for (auto &Inc : S.IncludeHeaders)
        remapRef(Inc.IncludeHeader, Mappings, Saver);
      Builder.insert(S);
    }
    Data.Symbols = std::move(Builder).build();
  }

  if (Data.Refs) {
    RefSlab::Builder Builder;
    for (const auto &Entry : *Data.Refs) {
      for (const auto &R : Entry.second) {
        Ref MR = R; // mutable copy
        remapCharURI(MR.Location.FileURI, Mappings, Saver);
        Builder.insert(Entry.first, MR);
      }
    }
    Data.Refs = std::move(Builder).build();
  }

  // We must rebuild the StringMap because keys may change.  All StringRef
  // fields (URI, DirectIncludes) are saved into Saver because the old
  // StringMap is destroyed below.
  if (Data.Sources) {
    IncludeGraph NewSources;
    for (auto &Entry : *Data.Sources) {
      IncludeGraphNode IGN = Entry.getValue();
      remapOrCopyRef(IGN.URI, Mappings, Saver);
      for (auto &Inc : IGN.DirectIncludes)
        remapOrCopyRef(Inc, Mappings, Saver);
      NewSources[IGN.URI] = std::move(IGN);
    }
    Data.Sources = std::move(NewSources);
  }

  if (Data.Cmd) {
    remapStdStr(Data.Cmd->Directory, Mappings);
    for (auto &Arg : Data.Cmd->CommandLine)
      remapStdStr(Arg, Mappings);
    remapStdStr(Data.Cmd->Filename, Mappings);
  }
}

} // namespace
} // namespace clangd
} // namespace clang

int main(int Argc, const char **Argv) {
  using namespace clang::clangd;

  llvm::sys::PrintStackTraceOnErrorSignal(Argv[0]);
  llvm::cl::HideUnrelatedOptions(RemapCategory);
  llvm::cl::ParseCommandLineOptions(Argc, Argv,
                                    "clangd-remap: rewrite paths inside "
                                    "background-index .idx shards\n");

  StreamLogger Logger(llvm::errs(), LogLevel);
  LoggingSession LoggingSession(Logger);

  auto Mappings = parsePathMappings(PathMappingsArg);
  if (!Mappings) {
    elog("Invalid --path-mappings: {0}", Mappings.takeError());
    return 1;
  }
  if (Mappings->empty()) {
    elog("No path mappings specified.");
    return 1;
  }

  // Gather all shard files from the index directory.
  auto AllShards = collectShards(IndexDir);
  if (AllShards.empty()) {
    log("No .idx files found in the specified directories.");
    return 0;
  }

  log("Found {0} shard(s) to process.", AllShards.size());
  for (const auto &M : *Mappings)
    log("  Path mapping: {0}", M);

  if (NumThreads.getValue() != 0)
    llvm::parallel::strategy = llvm::hardware_concurrency(NumThreads);

  std::atomic<unsigned> Errors{0};
  std::atomic<unsigned> FilesRenamed{0};
  std::atomic<unsigned> FilesUnchanged{0};

  llvm::parallelFor(0, AllShards.size(), [&](size_t I) {
    const std::string &ShardPath = AllShards[I];

    auto Buf = llvm::MemoryBuffer::getFile(ShardPath);
    if (!Buf) {
      elog("Cannot read {0}: {1}", ShardPath, Buf.getError().message());
      ++Errors;
      return;
    }

    auto Parsed = readIndexFile((*Buf)->getBuffer(), SymbolOrigin::Background);
    if (!Parsed) {
      elog("Cannot parse {0}: {1}", ShardPath, Parsed.takeError());
      ++Errors;
      return;
    }

    // Derive the new shard filename before remapping, so we can match
    // against original (un-remapped) source URIs.
    llvm::StringRef OldFilename = llvm::sys::path::filename(ShardPath);
    std::string NewFilename =
        deriveNewFilename(*Parsed, OldFilename, *Mappings);

    // Remap all paths in the parsed data
    llvm::BumpPtrAllocator Arena;
    llvm::StringSaver Saver(Arena);
    remapIndexData(*Parsed, *Mappings, Saver);

    // Write the remapped shard (possibly under a new name)
    llvm::StringRef ParentDir = llvm::sys::path::parent_path(ShardPath);
    llvm::SmallString<256> NewPath(ParentDir);
    llvm::sys::path::append(NewPath, NewFilename);
    if (auto Err = llvm::writeToOutput(NewPath, [&](llvm::raw_ostream &OS) {
          IndexFileOut Out(*Parsed);
          Out.Format = IndexFileFormat::RIFF;
          OS << Out;
          return llvm::Error::success();
        })) {
      elog("Cannot write {0}: {1}", NewPath, std::move(Err));
      ++Errors;
      return;
    }

    // If the filename changed, remove the old shard
    if (NewFilename != OldFilename) {
      llvm::sys::fs::remove(ShardPath);
      ++FilesRenamed;
    } else
      ++FilesUnchanged;
  });

  unsigned Renamed = FilesRenamed.load();
  unsigned Unchanged = FilesUnchanged.load();
  log("Processed: {0} shard(s), {1} renamed, {2} unchanged, {3} error(s).",
      Renamed + Unchanged, Renamed, Unchanged, Errors.load());
  return Errors.load() > 0 ? 1 : 0;
}
