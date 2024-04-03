//===--- FileDistance.cpp - File contents container -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "URI.h"
#include "support/Logger.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Path.h"
#include <queue>

namespace clang {
namespace clangd {

// Convert a path into the canonical form.
// Canonical form is either "/", or "/segment" * N:
//   C:\foo\bar --> /c:/foo/bar
//   /foo/      --> /foo
//   a/b/c      --> /a/b/c
static llvm::SmallString<128> canonicalize(llvm::StringRef Path) {
  llvm::SmallString<128> Result = Path.rtrim('/');
  native(Result, llvm::sys::path::Style::posix);
  if (Result.empty() || Result.front() != '/')
    Result.insert(Result.begin(), '/');
  return Result;
}

constexpr const unsigned FileDistance::Unreachable;
const llvm::hash_code FileDistance::RootHash =
    llvm::hash_value(llvm::StringRef("/"));

FileDistance::FileDistance(llvm::StringMap<SourceParams> Sources,
                           const FileDistanceOptions &Opts)
    : Opts(Opts) {
  llvm::DenseMap<llvm::hash_code, llvm::SmallVector<llvm::hash_code>> DownEdges;
  // Compute the best distance following only up edges.
  // Keep track of down edges, in case we can use them to improve on this.
  for (const auto &S : Sources) {
    auto Canonical = canonicalize(S.getKey());
    dlog("Source {0} = {1}, MaxUp = {2}", Canonical, S.second.Cost,
         S.second.MaxUpTraversals);
    // Walk up to ancestors of this source, assigning cost.
    llvm::StringRef Rest = Canonical;
    llvm::hash_code Hash = llvm::hash_value(Rest);
    for (unsigned I = 0; !Rest.empty(); ++I) {
      Rest = parent_path(Rest, llvm::sys::path::Style::posix);
      auto NextHash = llvm::hash_value(Rest);
      auto &Down = DownEdges[NextHash];
      if (!llvm::is_contained(Down, Hash))
        Down.push_back(Hash);
      // We can't just break after MaxUpTraversals, must still set DownEdges.
      if (I > S.getValue().MaxUpTraversals) {
        if (Cache.contains(Hash))
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
  std::queue<llvm::hash_code> Next;
  for (auto Child : DownEdges.lookup(llvm::hash_value(llvm::StringRef(""))))
    Next.push(Child);
  while (!Next.empty()) {
    auto Parent = Next.front();
    Next.pop();
    auto ParentCost = Cache.lookup(Parent);
    for (auto Child : DownEdges.lookup(Parent)) {
      if (Parent != RootHash || Opts.AllowDownTraversalFromRoot) {
        auto &ChildCost =
            Cache.try_emplace(Child, Unreachable).first->getSecond();
        if (ParentCost + Opts.DownCost < ChildCost)
          ChildCost = ParentCost + Opts.DownCost;
      }
      Next.push(Child);
    }
  }
}

unsigned FileDistance::distance(llvm::StringRef Path) {
  auto Canonical = canonicalize(Path);
  unsigned Cost = Unreachable;
  llvm::SmallVector<llvm::hash_code> Ancestors;
  // Walk up ancestors until we find a path we know the distance for.
  for (llvm::StringRef Rest = Canonical; !Rest.empty();
       Rest = parent_path(Rest, llvm::sys::path::Style::posix)) {
    auto Hash = llvm::hash_value(Rest);
    if (Hash == RootHash && !Ancestors.empty() &&
        !Opts.AllowDownTraversalFromRoot) {
      Cost = Unreachable;
      break;
    }
    auto It = Cache.find(Hash);
    if (It != Cache.end()) {
      Cost = It->second;
      break;
    }
    Ancestors.push_back(Hash);
  }
  // Now we know the costs for (known node, queried node].
  // Fill these in, walking down the directory tree.
  for (llvm::hash_code Hash : llvm::reverse(Ancestors)) {
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
        llvm::consumeError(U.takeError());
    }
    dlog("FileDistance for scheme {0}: {1}/{2} sources", Scheme,
         SchemeSources.size(), Sources.size());
    Delegate.reset(new FileDistance(std::move(SchemeSources), Opts));
  }
  return *Delegate;
}

static std::pair<std::string, int> scopeToPath(llvm::StringRef Scope) {
  llvm::SmallVector<llvm::StringRef> Split;
  Scope.split(Split, "::", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  return {"/" + llvm::join(Split, "/"), Split.size()};
}

static FileDistance
createScopeFileDistance(llvm::ArrayRef<std::string> QueryScopes) {
  FileDistanceOptions Opts;
  Opts.UpCost = 2;
  Opts.DownCost = 4;
  Opts.AllowDownTraversalFromRoot = false;

  llvm::StringMap<SourceParams> Sources;
  llvm::StringRef Preferred =
      QueryScopes.empty() ? "" : QueryScopes.front().c_str();
  for (llvm::StringRef S : QueryScopes) {
    SourceParams Param;
    // Penalize the global scope even it's preferred, as all projects can define
    // symbols in it, and there is pattern where using-namespace is used in
    // place of enclosing namespaces (e.g. in implementation files).
    if (S == Preferred)
      Param.Cost = S == "" ? 4 : 0;
    else if (Preferred.starts_with(S) && !S.empty())
      continue; // just rely on up-traversals.
    else
      Param.Cost = S == "" ? 6 : 2;
    auto Path = scopeToPath(S);
    // The global namespace is not 'near' its children.
    Param.MaxUpTraversals = std::max(Path.second - 1, 0);
    Sources[Path.first] = std::move(Param);
  }
  return FileDistance(std::move(Sources), Opts);
}

ScopeDistance::ScopeDistance(llvm::ArrayRef<std::string> QueryScopes)
    : Distance(createScopeFileDistance(QueryScopes)) {}

unsigned ScopeDistance::distance(llvm::StringRef SymbolScope) {
  return Distance.distance(scopeToPath(SymbolScope).first);
}

} // namespace clangd
} // namespace clang
