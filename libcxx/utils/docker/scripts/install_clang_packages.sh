#!/usr/bin/env bash
#===- libcxx/utils/docker/scripts/install_clang_package.sh -----------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===-----------------------------------------------------------------------===//

set -e

function show_usage() {
  cat << EOF
Usage: install_clang_package.sh [options]

Install
Available options:
  -h|--help           show this help message
  --version           the numeric version of the package to use.
EOF
}

VERSION="9"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      shift
      VERSION="$1"
      shift
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
  esac
done

set -x

curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
add-apt-repository -s "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs) main"
apt-get update
apt-get upgrade -y
apt-get install -y --no-install-recommends "clang-$VERSION"

# FIXME(EricWF): Remove this once the clang packages are no longer broken.
if [ -f "/usr/local/bin/clang" ]; then
  echo "clang already exists"
  exit 1
else
  CC_BINARY="$(which clang-$VERSION)"
  ln -s "$CC_BINARY" "/usr/local/bin/clang"
fi
if [ -f "/usr/local/bin/clang++" ]; then
  echo "clang++ already exists"
  exit 1
else
  CXX_BINARY="$(which clang++-$VERSION)"
  ln -s "$CXX_BINARY" "/usr/local/bin/clang++"
fi

echo "Testing clang version..."
clang --version

echo "Testing clang++ version..."
clang++ --version

# Figure out the libc++ and libc++abi package versions that we want.
if [ "$VERSION" == "" ]; then
  VERSION="$(apt-cache search 'libc\+\+-[0-9]-dev' | awk '{print $1}' | awk -F- '{print $2}')"
  echo "Installing version '$VERSION'"
fi

apt-get purge -y "libc++-$VERSION-dev" "libc++abi-$VERSION-dev"
apt-get install -y --no-install-recommends "libc++-$VERSION-dev" "libc++abi-$VERSION-dev"

echo "Done"
