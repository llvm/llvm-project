#!/bin/bash

#| Usage: translate-legacy-branch-to-omnibus-monorepo.sh SOURCE-REPO-KIND BASE-BRANCH TARGET-REPO-DIR
#|
#| Wrapper for git-filter-repo (https://github.com/newren/git-filter-repo) and
#| git-rebase to assist with rebasing branches from legacy ROCm repositories
#| into the new "omnibus" llvm-project monorepo.
#|
#| Must be executed from within the source repo, with the branch to be
#| translated already checked out.
#|
#| SOURCE-REPO-KIND identifies which repo is being translated:
#|
#|   Value        Repository
#|   -----------  ---------------------------------------------------------
#|   comgr        https://github.com/RadeonOpenCompute/ROCm-CompilerSupport
#|   device-libs  https://github.com/RadeonOpenCompute/ROCm-Device-Libs
#|   hipcc        https://github.com/ROCm-Developer-Tools/HIPCC
#|
#| BASE-BRANCH identifies the branch upon which the to-be-translated branch is
#| based (e.g. amd-stg-open).
#|
#| The TARGET-REPO-DIR must locate a fresh clone of the ROCm llvm-project
#| repository (https://github.com/RadeonOpenCompute/llvm-project). If the
#| script is to be invoked repeatedly the same clone may be reused. It is
#| NOT recommended to use any clone which contains local refs that have not
#| been saved elsewhere, as they may be clobbered!
#|
#| The currently checked out branch from the source repo is translated into the
#| target repo, and can be used to e.g. create pull-requests from the target
#| repo as usual.

set -euo pipefail

usage() {
  >&2 sed -n 's/^#| \?\(.*\)$/\1/p' "$0"
}

request_git_filter_repo() {
  >&2 printf 'error: the git-filter-repo command is not present in PATH\n'
  >&2 printf '  ...: it can be downloaded to your current directory via:\n'
  >&2 printf '  ...: curl -sSL https://github.com/newren/git-filter-repo/releases/download/v2.38.0/git-filter-repo-2.38.0.tar.xz | tar -xJf - --strip-components=1 git-filter-repo-2.38.0/git-filter-repo'
}

target_repo_wrong() {
  >&2 printf 'error: target repo is not an omnibus monorepo: %s\n' "$1"
}

unknown_source_repo_kind() {
  >&2 printf 'error: unknown source repo kind: %s\n' "$1"
  >&2 printf '  ...: legal values: comgr device-libs hipcc\n'
}

main() {
  hash git-filter-repo >/dev/null 2>&1 || { request_git_filter_repo; exit 1; }
  (( $# >= 3 )) || { usage; exit 2; }
  local source_repo_kind="$1"; shift
  local base_branch="$1"; shift
  local target_repo_path="$1"; shift

  local path
  case "$source_repo_kind" in
    comgr) path=lib/comgr/;;
    device-libs) path='';;
    hipcc) path='';;
    *) unknown_source_repo_kind "$source_repo_kind"; exit 3;;
  esac

  local topic_branch
  topic_branch="$(git rev-parse --abbrev-ref HEAD)"

  local topic_branch_size
  topic_branch_size="$(git rev-list --count ^"$base_branch" HEAD)"

  git-filter-repo \
    --target "$target_repo_path" \
    --replace-refs delete-no-add \
    --path "$path" \
    --path-rename "$path:amd/$source_repo_kind/" \
    --refs "$topic_branch" \
    --force
  git -C "$target_repo_path" rebase \
    "$topic_branch~$topic_branch_size" \
    "$topic_branch" \
    --onto="$base_branch"
}

main "$@"
