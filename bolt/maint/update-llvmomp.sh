#!/bin/bash

boltpath="$(pwd)"

# Switch to the llvmomp branch
git checkout llvmomp

# Get the latest llvm-project
if [ -d "${boltpath}/llvm-project" ]; then
  cd "${boltpath}/llvm-project"
  git pull
  git checkout master
else
  git clone https://github.com/llvm/llvm-project.git "${boltpath}/llvm-project"
fi

# Find the corresponding latest commit of the llvmomp branch
cd "$boltpath"
if [ x"$(git log HEAD~..HEAD | grep cherry-pick:)" = x ]; then
  # The last commit in llvm-mirror: d69d1aa131b4cf339bfac116e50da33a5f94b861
  commit_id_begin="d69d1aa131b4cf339bfac116e50da33a5f94b861"
else
  # Extract the first "cherry-pick:" comment from the latest log.
  commit_id_begin=$(git log HEAD~..HEAD | grep -Po "cherry-pick: \K[\w]+" | head -n 1)
fi
echo "last commit: $commit_id_begin"

rm -rf "${boltpath}/llvm-project/patch_diff.tmp"
rm -rf "${boltpath}/llvm-project/patch_log.tmp"
rm -rf "${boltpath}/llvm-project/patch_changed_list.tmp"

cd "${boltpath}/llvm-project/openmp"

for commit_id in $(git rev-list --reverse ${commit_id_begin}..HEAD); do
  echo "checking $commit_id"
  # Check if OpenMP part is changed.
  if [ x"$(git diff --name-only --relative ${commit_id}~..${commit_id})" != x ]; then
    echo "\n#########################\n"
    # Create a diff file
    git diff --relative ${commit_id}~..${commit_id} > "${boltpath}/llvm-project/patch_diff.tmp"
    # Create a list of changed files
    git diff --name-only --relative ${commit_id}~..${commit_id} > "${boltpath}/llvm-project/patch_changed_list.tmp"
    # Create a log file for this
    git log ${commit_id}~..${commit_id} --pretty=format:"%B" > "${boltpath}/llvm-project/patch_log.tmp"
    echo "" >> "${boltpath}/llvm-project/patch_log.tmp"
    echo "cherry-pick: $commit_id" >> "${boltpath}/llvm-project/patch_log.tmp"
    echo "https://github.com/llvm/llvm-project/commit/${commit_id}" >> "${boltpath}/llvm-project/patch_log.tmp"
    author_info="$(git show --pretty="%aN <%aE>" $commit_id | head -n 1)"
    timestamp_info="$(git show --pretty="%cd" $commit_id | head -n 1)"

    # Go to the BOLT repository
    cd "${boltpath}"

    # Apply the diff to the llvmomp branch
    git apply "${boltpath}/llvm-project/patch_diff.tmp"
    for changed_file in $(cat "${boltpath}/llvm-project/patch_changed_list.tmp"); do
      git add $changed_file
    done

    # Commit it.
    cat "${boltpath}/llvm-project/patch_log.tmp"
    git commit -F "${boltpath}/llvm-project/patch_log.tmp" --author="$author_info" --date="$timestamp_info"

    # Go to the llvm-project repository.
    cd "${boltpath}/llvm-project/openmp"
  fi
done

echo "complete"
