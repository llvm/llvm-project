//===--- FileDistance.cpp - File contents container -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The FileDistance structure allows calculating the minimum distance to paths
// in a single tree.
// We simply walk up the path's ancestors until we find a node whose cost is
// known, and add the cost of walking back down. Initialization ensures this
// gives the correct path to the roots.
// We cache the results, so that the runtime is O(|A|), where A is the set of
// all distinct ancestors of visited paths.
//
// Example after initialization with /=2, /bar=0, DownCost = 1:
//  / = 2
//    /bar = 0
//
// After querying /foo/bar and /bar/foo:
//  / = 2
//    /bar = 0
//      /bar/foo = 1
//    /foo = 3
//      /foo/bar = 4
//
// URIDistance creates FileDistance lazily for each URI scheme encountered. In
// practice this is a small constant factor.
//
//===-------------------------------------------------------------------------//

#include "FileDistance.h"
#include "Logger.h"
#include "llvm/ADT/STLExtras.h"
#include <queue>

namespace clang {
namespace clangd {
using namespace llvm;

// Convert a path into the canonical form.
// Canonical form is either "/", or "/segment" * N:
//   C:\foo\bar --> /c:/foo/bar
//   /foo/      --> /foo
//   a/b/c      --> /a/b/c
static SmallString<128> canonicalize(StringRef Path) {
  SmallString<128> Result = Path.rtrim('/');
  native(Result, sys::path::Style::posix);
  if (Result.empty() || Result.front() != '/')
    Result.insert(Result.begin(), '/');
  return Result;
}

constexpr const unsigned FileDistance::Unreachable;

FileDistance::FileDistance(StringMap<SourceParams> Sources,
                           const FileDistanceOptions &Opts)
    : Opts(Opts) {
  llvm::DenseMap<hash_code, SmallVector<hash_code, 4>> DownEdges;
  // Compute the best distance following only up edges.
  // Keep track of down edges, in case we can use them to improve on this.
  for (const auto &S : Sources) {
    auto Canonical = canonicalize(S.getKey());
    dlog("Source {0} = {1}, MaxUp = {2}", Canonical, S.second.Cost,
         S.second.MaxUpTraversals);
    // Walk up to ancestors of this source, assigning cost.
    StringRef Rest = Canonical;
    llvm::hash_code Hash = hash_value(Rest);
    for (unsigned I = 0; !Rest.empty(); ++I) {
      Rest = parent_path(Rest, sys::path::Style::posix);
      auto NextHash = hash_value(Rest);
      auto &Down = DownEdges[NextHash];
      if (std::find(Down.begin(), Down.end(), Hash) == Down.end())
        DownEdges[NextHash].push_back(Hash);
      // We can't just break after MaxUpTraversals, must still set DownEdges.
      if (I > S.getValue().MaxUpTraversals) {
        if (Cache.find(Hash) != Cache.end())
          break;
      } else {
        unsigned Cost = S.getValue().Cost + I * Opts.UpCost;
        auto R = Cache.try_emplace(Hash, Cost);
        if (!R.second) {
          if (Cost < R.first->second) {
            R.first->second = Cost;
          } else {
            // If we're not the best way to get to this path, stop assigning.
            break;
          }
        }
      }
      Hash = NextHash;
    }
  }
  // Now propagate scores parent -> child if that's an improvement.
  // BFS ensures we propagate down chains (must visit parents before children).
  std::queue<hash_code> Next;
  for (auto Child : DownEdges.lookup(hash_value(llvm::StringRef(""))))
    Next.push(Child);
  while (!Next.empty()) {
    auto ParentCost = Cache.lookup(Next.front());
    for (auto Child : DownEdges.lookup(Next.front())) {
      auto &ChildCost =
          Cache.try_emplace(Child, Unreachable).first->getSecond();
      if (ParentCost + Opts.DownCost < ChildCost)
        ChildCost = ParentCost + Opts.DownCost;
      Next.push(Child);
    }
    Next.pop();
  }
}

unsigned FileDistance::distance(StringRef Path) {
  auto Canonical = canonicalize(Path);
  unsigned Cost = Unreachable;
  SmallVector<hash_code, 16> Ancestors;
  // Walk up ancestors until we find a path we know the distance for.
  for (StringRef Rest = Canonical; !Rest.empty();
       Rest = parent_path(Rest, sys::path::Style::posix)) {
    auto Hash = hash_value(Rest);
    auto It = Cache.find(Hash);
    if (It != Cache.end()) {
      Cost = It->second;
      break;
    }
    Ancestors.push_back(Hash);
  }
  // Now we know the costs for (known node, queried node].
  // Fill these in, walking down the directory tree.
  for (hash_code Hash : reverse(Ancestors)) {
    if (Cost != Unreachable)
      Cost += Opts.DownCost;
    Cache.try_emplace(Hash, Cost);
  }
  dlog("distance({0} = {1})", Path, Cost);
  return Cost;
}

unsigned URIDistance::distance(llvm::StringRef URI) {
  auto R = Cache.try_emplace(llvm::hash_value(URI), FileDistance::Unreachable);
  if (!R.second)
    return R.first->getSecond();
  if (auto U = clangd::URI::parse(URI)) {
    dlog("distance({0} = {1})", URI, U->body());
    R.first->second = forScheme(U->scheme()).distance(U->body());
  } else {
    log("URIDistance::distance() of unparseable {0}: {1}", URI, U.takeError());
  }
  return R.first->second;
}

FileDistance &URIDistance::forScheme(llvm::StringRef Scheme) {
  auto &Delegate = ByScheme[Scheme];
  if (!Delegate) {
    llvm::StringMap<SourceParams> SchemeSources;
    for (const auto &Source : Sources) {
      if (auto U = clangd::URI::create(Source.getKey(), Scheme))
        SchemeSources.try_emplace(U->body(), Source.getValue());
      else
        consumeError(U.takeError());
    }
    dlog("FileDistance for scheme {0}: {1}/{2} sources", Scheme,
         SchemeSources.size(), Sources.size());
    Delegate.reset(new FileDistance(std::move(SchemeSources), Opts));
  }
  return *Delegate;
}

} // namespace clangd
} // namespace clang
