#!/usr/bin/env bash
set -x

if [[ "$1" == "--new" ]]; then
  shift
fi

# Update the libc++ sources in the image in order to use the most recent version of
# run_buildbots.sh
cd /libcxx
git pull
/libcxx/utils/docker/scripts/run_buildbot_new.sh "$@"
