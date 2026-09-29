# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

#
# This script determines whether the libc++ Buildkite pipeline should be triggered
# on a change. This is required because Buildkite gets notified for every PR in the
# LLVM monorepo, and we make it a no-op unless the libc++ pipeline needs to be triggered.
#

# Set by buildkite
: ${BUILDKITE_PULL_REQUEST_BASE_BRANCH:=}

# Fetch origin to have an up to date merge base for the diff.
git fetch origin
# List of files affected by this commit
: ${MODIFIED_FILES:=$(git diff --name-only origin/${BUILDKITE_PULL_REQUEST_BASE_BRANCH}...HEAD)}

echo "Files modified:" >&2
echo "$MODIFIED_FILES" >&2
modified_dirs=$(echo "$MODIFIED_FILES" | cut -d'/' -f1 | sort -u)
echo "Directories modified:" >&2
echo "$modified_dirs" >&2

# If libc++ or one of the runtimes directories changed, trigger the libc++ Buildkite pipeline.
if echo "$modified_dirs" | grep -q -E "^(libcxx|libcxxabi|libunwind|runtimes|cmake)$"; then
    buildkite-agent pipeline upload libcxx/utils/ci/buildkite-pipeline.yml
else
    echo "No Buildkite jobs to trigger"
fi
