#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

#
# This file generates a Buildkite pipeline that triggers the libc++ CI
# job(s) if needed. The intended usage of this script is to be piped
# into `buildkite-agent pipeline upload`.
#

if git diff --name-only HEAD~ | grep -q -E "libcxx/|libcxxabi/"; then
  skip="false"
else
  skip="The commit does not touch libc++ or libc++abi"
fi

reviewID="$(git log --format=%B -n 1 | sed -nE 's/^Review-ID:[[:space:]]*(.+)$/\1/p')"
if [[ "${reviewID}" != "" ]]; then
  buildMessage="https://llvm.org/${reviewID}"
else
  buildMessage="Push to branch ${BUILDKITE_BRANCH}"
fi

cat <<EOF
steps:
  - trigger: "libcxx-ci"
    async: true
    build:
      message: "${buildMessage}"
      commit: "${BUILDKITE_COMMIT}"
      branch: "${BUILDKITE_BRANCH}"
    skip: "${skip}"
EOF
