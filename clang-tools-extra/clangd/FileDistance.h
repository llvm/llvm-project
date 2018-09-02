//===--- FileDistance.h - File proximity scoring -----------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This library measures the distance between file paths.
// It's used for ranking symbols, e.g. in code completion.
//  |foo/bar.h -> foo/bar.h| = 0.
//  |foo/bar.h -> foo/baz.h| < |foo/bar.h -> baz.h|.
// This is an edit-distance, where edits go up or down the directory tree.
// It's not symmetrical, the costs of going up and down may not match.
//
// Dealing with multiple sources:
// In practice we care about the distance from a source file, but files near
// its main-header and #included files are considered "close".
// So we start with a set of (anchor, cost) pairs, and call the distance to a
// path the minimum of `cost + |source -> path|`.
//
// We allow each source to limit the number of up-traversals paths may start
// with. Up-traversals may reach things that are not "semantically near".
//
// Symbol URI schemes:
// Symbol locations may be represented by URIs rather than file paths directly.
// In this case we want to perform distance computations in URI space rather
// than in file-space, without performing redundant conversions.
// Therefore we have a lookup structure that accepts URIs, so that intermediate
// calculations for the same scheme can be reused.
//
// Caveats:
// Assuming up and down traversals each have uniform costs is simplistic.
// Often there are "semantic roots" whose children are almost unrelated.
// (e.g. /usr/include/, or / in an umbrella repository). We ignore this.
//
//===----------------------------------------------------------------------===//

#include "URI.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"

namespace clang {
namespace clangd {

struct FileDistanceOptions {
  unsigned UpCost = 2;      // |foo/bar.h -> foo|
  unsigned DownCost = 1;    // |foo -> foo/bar.h|
  unsigned IncludeCost = 2; // |foo.cc -> included_header.h|
};

struct SourceParams {
  // Base cost for paths starting at this source.
  unsigned Cost = 0;
  // Limits the number of upwards traversals allowed from this source.
  unsigned MaxUpTraversals = std::numeric_limits<unsigned>::max();
};

// Supports lookups to find the minimum distance to a file from any source.
// This object should be reused, it memoizes intermediate computations.
class FileDistance {
public:
  static constexpr unsigned Unreachable = std::numeric_limits<unsigned>::max();

  FileDistance(llvm::StringMap<SourceParams> Sources,
               const FileDistanceOptions &Opts = {});

  // Computes the minimum distance from any source to the file path.
  unsigned distance(llvm::StringRef Path);

private:
  // Costs computed so far. Always contains sources and their ancestors.
  // We store hash codes only. Collisions are rare and consequences aren't dire.
  llvm::DenseMap<llvm::hash_code, unsigned> Cache;
  FileDistanceOptions Opts;
};

// Supports lookups like FileDistance, but the lookup keys are URIs.
// We convert each of the sources to the scheme of the URI and do a FileDistance
// comparison on the bodies.
class URIDistance {
public:
  URIDistance(llvm::StringMap<SourceParams> Sources,
              const FileDistanceOptions &Opts = {})
      : Sources(Sources), Opts(Opts) {}

  // Computes the minimum distance from any source to the URI.
  // Only sources that can be mapped into the URI's scheme are considered.
  unsigned distance(llvm::StringRef URI);

private:
  // Returns the FileDistance for a URI scheme, creating it if needed.
  FileDistance &forScheme(llvm::StringRef Scheme);

  // We cache the results using the original strings so we can skip URI parsing.
  llvm::DenseMap<llvm::hash_code, unsigned> Cache;
  llvm::StringMap<SourceParams> Sources;
  llvm::StringMap<std::unique_ptr<FileDistance>> ByScheme;
  FileDistanceOptions Opts;
};

} // namespace clangd
} // namespace clang
