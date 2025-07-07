#!/usr/bin/env bash

WORKSPACE=$PWD
cd "${WORKSPACE}/llvm-project" || exit 1

set -x
echo "git push origin HEAD:refs/heads/upstream-main"
git push origin HEAD:refs/heads/upstream-main
if [ "$?" -eq 0 ]; then
        echo "********************************************************"
        echo "Execution of command : git push origin HEAD:refs/heads/upstream-main - was successful" >> "${WORKSPACE}/command_executed.txt"
        echo "Execution of command : git push origin HEAD:refs/heads/upstream-main - was successful"
        echo "********************************************************"
else
        echo "########################################################"
        echo "Execution of command : git push origin HEAD:refs/heads/upstream-main - was failed"
        echo "Please check ${WORKSPACE}/command_executed.txt file for commands executed till now"
        rm -v "${WORKSPACE}"/build_success.txt
        exit 1
        echo "########################################################"
fi

if [ ! -f "${WORKSPACE}/build_success.txt" ]
then
   echo "Job failed - Please check ${WORKSPACE}/command_executed.txt file for commands executed till now"
   exit 1
fi

cd "${WORKSPACE}/llvm-project" || exit 1
declare -A branch_map
branch_map["release/20.x"]="aocc/upstream/20.x"
branch_map["main"]="aocc/upstream/main"

for src_branch in "${!branch_map[@]}"; do
    dest_branch="${branch_map[$src_branch]}"
    echo "Pushing branch from $src_branch to $dest_branch"
    if ! git fetch origin $src_branch:$dest_branch; then
        echo "Failed to fetch branch: $src_branch"
    fi
    if ! git push origin "$dest_branch:$dest_branch"; then
        echo "Failed to push branch: $dest_branch"
    fi
done
