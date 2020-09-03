# Apple's branching scheme for llvm-project

There are currently three namespaces for branches on
[github.com/apple/llvm-project](https://github.com/apple/llvm-project):

 1. `llvm.org/*`, for forwarded branches from
    [github.com/llvm](https://github.com/llvm/llvm-project);
 2. `apple/*`, for standalone downstream content; and
 3. `swift/*`, for downstream content that depends on
    [Swift](https://github.com/apple/swift).

## Forwarded branches from [github.com/llvm](https://github.com/llvm/llvm-project)

The `llvm.org/*` branches are forwarded, unchanged, from
[github.com/llvm/llvm-project](https://github.com/llvm/llvm-project).  These
are read-only, exact copies of the upstream LLVM project's branches.  They are
forwarded here as a convenience for easy reference, to avoid the need for extra
remotes.

- [llvm.org/master](https://github.com/apple/llvm-project/tree/llvm.org/master)
  is the most important branch here, matching the LLVM project's
  [master](https://github.com/llvm/llvm-project/tree/master) branch.

## Downstream branches that are standalone

The `apple/*` branches have downstream content, besides what is in the LLVM
project.  This content includes some patches that have not yet been fully
upstreamed to the LLVM project, including some special support for Swift.
Critically, however, none of these branches *depend on* the
[github.com/apple/swift](https://github.com/apple/swift) repository.

Today there are a few non-trivial differences from LLVM, but we are
actively working on either upstreaming or reverting those differences. The goal
is to fully eliminate all differences between `apple/master` and
`llvm.org/master`.

Any LLVM development that does not depend on the Swift repository should happen
upstream. The only changes that are allowed to be submitted without going
through upstream LLVM are those that are either directly related to upstreaming
content or that are needed because of the existing differences (e.g., resolving
merge conflicts or fixing build errors).

- [apple/master](https://github.com/apple/llvm-project/tree/apple/master) is
  directly downstream of
  [llvm.org/master](https://github.com/apple/llvm-project/tree/llvm.org/master).
  There is a gated automerger that does testing before merging in.  Most
  changes to this branch should be redirected to <https://reviews.llvm.org/>
  (see also <http://llvm.org/docs/Contributing.html>).
- `apple/stable/*`: These branches are periodic stabilization branches, where
  fixes are cherry-picked from LLVM.  At time of writing:
    - [apple/stable/20200108](https://github.com/apple/llvm-project/tree/apple/stable/20200108)
      is the most recent stabilization branch.
    - [apple/stable/20190619](https://github.com/apple/llvm-project/tree/apple/stable/20190619)
      is the current stabilization branch for
      [swift/master](https://github.com/apple/llvm-project/tree/swift/master)
      (see below).

## Downstream branches that depend on [Swift](https://github.com/apple/swift)

The `swift/*` branches are downstream of `apple/*`, and include content that
depends on [Swift](https://github.com/apple/swift).  The naming scheme is
`swift/<swift-branch>`, where `<swift-branch>` is the aligned Swift branch.

The branches are automerged from a branch in the `apple/*` namespace.  They are
expected to have zero differences outside the `lldb/` and `apple-llvm-config/`
directories. Any changes outside of these directories should be submitted in
the upstream LLVM repository.

These are the most important branches:

- [swift/master-next](https://github.com/apple/llvm-project/tree/swift/master-next)
  is downstream of
  [apple/master](https://github.com/apple/llvm-project/tree/apple/master) and
  aligned with Swift's
  [master-next](https://github.com/apple/swift/tree/master-next) branch.
- [swift/master](https://github.com/apple/llvm-project/tree/swift/master) is
  downstream of a stabilization branch in `apple/stable/*`
  ([apple/stable/20190619](https://github.com/apple/llvm-project/tree/apple/stable/20190619),
  as of time of writing) and aligned with Swift's
  [master](https://github.com/apple/swift/tree/master) branch.

## Historical trivia: mappings to branches from before the monorepo transition

Before the LLVM project's monorepo transition, Apple maintained downstream
forks of various split repositories.  Here is a mapping from a few of the new
branches in the llvm-project monorepo to their original split repositories.

- [apple/master](https://github.com/apple/llvm-project/tree/apple/master) was
  generated from the `upstream-with-swift` branches in
  [swift-clang](https://github.com/apple/swift-clang/),
  [swift-llvm](https://github.com/apple/swift-llvm/),
  [swift-compiler-rt](https://github.com/apple/swift-compiler-rt/),
  [swift-clang-tools-extra](https://github.com/apple/swift-clang-tools-extra/),
  and [swift-libcxx](https://github.com/apple/swift-libcxx/), with the notable
  **exclusion** of [swift-lldb](https://github.com/apple/swift-lldb/),
- [swift/master-next](https://github.com/apple/llvm-project/tree/swift/master-next)
  was generated from the `upstream-with-swift` branch in
  [swift-lldb](https://github.com/apple/swift-lldb/), interleaved with merges
  from [apple/master](https://github.com/apple/llvm-project/tree/apple/master).
- [apple/stable/20190104](https://github.com/apple/llvm-project/tree/apple/stable/20190104)
  was generated from the `swift-5.1-branch` branches in
  [swift-clang](https://github.com/apple/swift-clang/),
  [swift-llvm](https://github.com/apple/swift-llvm/),
  [swift-compiler-rt](https://github.com/apple/swift-compiler-rt/),
  [swift-clang-tools-extra](https://github.com/apple/swift-clang-tools-extra/),
  and [swift-libcxx](https://github.com/apple/swift-libcxx/), with the notable
  **exclusion** of [swift-lldb](https://github.com/apple/swift-lldb/),
- [swift/swift-5.1-branch](https://github.com/apple/llvm-project/tree/swift/swift-5.1-branch)
  was generated from the `swift-5.1-branch` branch in
  [swift-lldb](https://github.com/apple/swift-lldb/), interleaved with merges
  from
  [apple/stable/20190104](https://github.com/apple/llvm-project/tree/apple/stable/20190104).
- [swift/master](https://github.com/apple/llvm-project/tree/swift/master) was
  generated from the `stable` branch from all six split repos.
