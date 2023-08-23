#!/bin/bash

# Merges lightning/ec/device-libs, lightning/ec/support, and compute/ec/hipcc
# into lightning/ec/llvm-project.
#
# This script can optionally be called with one argument to "dispatch" to a
# specific "subcommand" function. For example, to update the in.d cache
# and invalidate (delete) the contents of out.d run:
#
# ~/omnibus$ ./omnibus.sh pull_ind_clobber_outd
#
# Or, to push the merged branches int out.d/llvm-project to gerrit-git.amd.com
# using a prefix of $GERRIT_USER/omnibus/, run:
#
# ~/omnibus$ ./omnibus.sh push_outd
#
# The default "main" function creates directories named in.d and out.d, cloning
# "input" repos into in.d to serve as local caches for gerrit-git.amd.com
# repos, and producing a merged result repo named out.d/llvm-project
#
# Example usage:
#
#   ~/omnibus$ ls
#   omnibus.sh
#   ~/omnibus$ ./omnibus.sh
#   ...
#   ~/omnibus$ git -C out.d/llvm-project log --oneline | head -1
#   25f71328f775 Merge branch hipcc/amd-stg-open into amd-stg-open
#   ~/omnibus$ ls out.d/llvm-project/amd
#   comgr  device-libs  hipcc
#
# The new "amd" top-level directory in the resulting llvm-project repository
# contains all of the sources we don't intend to upstream in the near term.
# Each subdirectory under "amd" can be built independently or be included in
# the enclosing LLVM build via the LLVM_EXTERNAL_PROJECTS mechanism.
#
# A note on mixing sources:
#
# With the move to keeping more sources in a single git repo, any workflow
# which depends on mixing sources from different logical branches (e.g.
# compiling upstream main LLVM and then compiling amd-stg-open
# device-libs/comgr against it) needs to be adapted. Git worktrees can achieve
# this without requiring a separate clone of the entire llvm-project repo; for
# example, to build with upstream main LLVM sources and amd-stg-open
# device-libs+comgr sources a worktree with amd-stg-open checked out can be
# created and referred to from a build of the main worktree:
#
#   ~/omnibus/out.d/llvm-project$ git checkout main
#   ~/omnibus/out.d/llvm-project$ git worktree add amd-stg-open
#   ~/omnibus/out.d/llvm-project$ cd build
#   ~/omnibus/out.d/llvm-project/build$ cmake \
#     -DLLVM_ENABLE_PROJECTS='llvm;clang;lld' \
#     -DLLVM_EXTERNAL_PROJECTS='devicelibs;comgr'
#     -DLLVM_EXTERNAL_DEVICELIBS_SOURCE_DIR="$(pwd)/../amd-stg-open/amd/device-libs" \
#     -DLLVM_EXTERNAL_COMGR_SOURCE_DIR="$(pwd)/../amd-stg-open/amd/comgr" \
#     ../llvm

set -euo pipefail
shopt -s nullglob extglob

# Configuration
readonly SUBPROJECTS=(device-libs comgr hipcc)
readonly BRANCHES=(amd-stg-open)
readonly GERRIT_USER="${GERRIT_USER:-$USER}"
readonly GERRIT_HOST=
readonly GERRIT_PORT=
readonly GERRIT_SSH="ssh://$GERRIT_USER@$GERRIT_HOST:$GERRIT_PORT"

# Functions (not intended for external use directly)

gerrit_query() {
  ssh -p $GERRIT_PORT "$GERRIT_USER"@$GERRIT_HOST gerrit query --format JSON "$@"
}

subproject_path() {
  local project=$1
  local prefix repo
  [[ $project == hipcc ]] && prefix=compute || prefix=lightning
  [[ $project == comgr ]] && repo=support || repo=$project
  printf '%s/ec/%s' "$prefix" "$repo"
}

populate_ind() {
  mkdir in.d
  for project in llvm-project "${SUBPROJECTS[@]}"; do
    [[ -d in.d/$project ]] && continue
    git clone "$GERRIT_SSH/$(subproject_path "$project")" "in.d/$project"
  done
}

reset_local_branches() {
  for project in llvm-project "${SUBPROJECTS[@]}"; do
    git -C "$1/$project" checkout --detach
    git -C "$1/$project" for-each-ref \
        --format '%(refname:short)' \
        refs/heads/ \
      | xargs --no-run-if-empty git -C "$1/$project" branch -D
    for branch in "${BRANCHES[@]}"; do
      git -C "$1/$project" branch --no-track "$branch" "origin/$branch"
    done
  done
}

filter_repo() {
  local source_dir=$1; shift
  local source_project=$1; shift
  local target_dir=$1; shift
  local target_project=$1; shift
  local path=''
  [[ $project == comgr ]] && path=lib/comgr/
  git-filter-repo \
    --source "$source_dir/$source_project/" \
    --target "$target_dir/$target_project/" \
    --force --replace-refs delete-no-add \
    --path "$path" --path-rename "$path:amd/$project/" "$@"
}

populate_outd() {
  mkdir out.d
  for project in llvm-project "${SUBPROJECTS[@]}"; do
    cp -r "in.d/$project" out.d/
  done
  for project in "${SUBPROJECTS[@]}"; do
    filter_repo in.d "$project" out.d "$project"
  done
  scp -p -P "$GERRIT_PORT" "$GERRIT_USER"@"$GERRIT_HOST":hooks/commit-msg \
    out.d/llvm-project/.git/hooks/
}

reset_remotes_outd() {
  for project in "${SUBPROJECTS[@]}"; do
    git -C out.d/llvm-project remote rm "$project" 2>/dev/null ||:
    git -C out.d/llvm-project remote add "$project" "../$project"
    git -C out.d/llvm-project fetch "$project"
  done
}

octopus_merge() {
  local -r branch=$1; shift
  local parents tree parent_args commit

  # Ideally this would be an actual octopus merge, but that requires forcing
  # these to have a common ancestor, which I can't seem to do without causing a
  # significant amount of pain during later merges. Even if it is possible with
  # a reasonable amount of effort, llvm-project already has multiple "root"
  # commits, so there doesn't seem to be much value in avoiding adding more.
  # So, we instead do this "manually", as we know by construction there are no
  # true merge conflicts, as all of the contents of each repo are in distinct
  # subdirectories.
  #
  # First, we checkout the branch we will merge into, and note it as the first
  # parent of the octopus we will create.
  git -C out.d/llvm-project checkout "$branch"
  parents=("$(git -C out.d/llvm-project rev-parse HEAD)")
  # Then, we complete the separate merges which we know will succeed.
  for project in "${SUBPROJECTS[@]}"; do
    git -C out.d/llvm-project merge --allow-unrelated-histories --no-ff \
      -m "Merge branch $project/$branch into $branch" \
      "$project/$branch"
  done
  # Now, we want to implement the conflict-free octopus merge with these
  # unrelated-history merges, so we first note the second-parent of each merge.
  for merge in $(git -C out.d/llvm-project rev-list --first-parent HEAD~"${#SUBPROJECTS[@]}"..HEAD); do
    parents+=("$(git -C out.d/llvm-project rev-parse "$merge"^2)")
  done
  # We know the contents of the tree, index, and the top commit are all the
  # actual contents we want in the octopus merge, so we just want to back up the
  # HEAD but leave the tree and index alone.
  git -C out.d/llvm-project reset --soft "$branch"
  # Now we can actually create the tree object for our octopus using the index...
  tree="$(git -C out.d/llvm-project write-tree)"
  # ...and create the commit using the tree and our known parents.
  parent_args=()
  for parent in "${parents[@]}"; do
    parent_args+=(-p "$parent")
  done
  commit="$(git -C out.d/llvm-project commit-tree "$tree" "${parent_args[@]}" \
    -m "Merge ${SUBPROJECTS[*]} into llvm-project")"
  # Finally we can point the branch itself at our newly created octopus merge.
  git -C out.d/llvm-project reset --hard "$commit"
  git -C out.d/llvm-project commit --amend --no-edit
}

populate_transferd() {
  mkdir transfer.d

  for project in llvm-project "${SUBPROJECTS[@]}"; do
    cp -r "in.d/$project" transfer.d/
  done
}

ensure_ind() {
  [[ -d in.d ]] || {
    populate_ind
    reset_local_branches in.d
    rm -rf out.d
  }
}

# "Subcommand" functions

merge_projects() {
  ensure_ind
  [[ -d out.d ]] || populate_outd
  reset_local_branches out.d
  reset_remotes_outd
  for branch in "${BRANCHES[@]}"; do
    octopus_merge "$branch"
  done
}

pull_ind_clobber_outd() {
  for project in llvm-project "${SUBPROJECTS[@]}"; do
    git -C "in.d/$project" fetch --all
  done
  reset_local_branches in.d
  rm -rf out.d
}

push_outd() {
  for branch in "${BRANCHES[@]}"; do
    git -C out.d/llvm-project push --force -o skip-validation origin \
      "$branch:$GERRIT_USER/omnibus/$branch"
  done
}

transfer_gerrit_patches() {
  ensure_ind
  [[ -d transfer.d ]] || populate_transferd
  local done_refs=()
  local failed_refs=()
  for project in "${SUBPROJECTS[@]}"; do
    for branch in "${BRANCHES[@]}"; do
      local refs
      local unpushed_refs=()
      refs=$(gerrit_query --current-patch-set \
        "status:open owner:self project:$(subproject_path "$project") branch:$branch" \
        | jq -r 'select(.type != "stats") |
                 "refs/changes/\(.number | tostring | .[-2:])/\(.number)/\(.currentPatchSet.number)"')
      for ref in $refs; do
        git -C "transfer.d/$project" fetch "$GERRIT_SSH/$(subproject_path "$project")" "$ref"
        git -C "transfer.d/$project" checkout -b "$(printf '%s' "$ref" | tr / _)" FETCH_HEAD
      done
      [[ -z "$refs" ]] || {
        refs="$(printf '%s' "$refs" | tr / _)"
        # shellcheck disable=SC2086
        filter_repo transfer.d "$project" transfer.d llvm-project --refs $refs
        for ref in $refs; do
          git -C transfer.d/llvm-project checkout -b "transfer_$ref" "$branch"
          if git -C transfer.d/llvm-project cherry-pick "$ref"; then
            unpushed_refs+=("transfer_$ref")
          else
            failed_refs+=("$ref")
            printf "Unable to cherry-pick %s! This script doesn't support patch series which depend on each other, so that might be why. Sorry, but you're on your own with this one. Skipping!\n" "$ref"
            git -C transfer.d/llvm-project cherry-pick --abort
          fi
        done
      }
      for ref in "${unpushed_refs[@]}"; do
        git -C transfer.d/llvm-project push origin "$ref:refs/for/$branch"
        done_refs+=("$ref")
      done
    done
  done
  (( ${#done_refs[@]} )) && {
    printf '\nThe transfer succeeded for the following refs:\n'
    printf '%s\n' "${done_refs[@]}"
  }
  (( ${#failed_refs[@]} )) && {
    printf '\nThe transfer failed for the following refs, which will need manual attention:\n'
    printf '%s\n' "${failed_refs[@]}"
  }
}

main() {
  merge_projects
  #transfer_gerrit_patches
}

(($#)) || set -- main

"$@"
