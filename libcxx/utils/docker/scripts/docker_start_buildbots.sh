#!/usr/bin/env bash
set -x

# Update the libc++ sources in the image in order to use the most recent version of
# run_buildbots.sh
cd /libcxx
git pull
source utils/docker/scripts/run_buildbots.sh "$@"
