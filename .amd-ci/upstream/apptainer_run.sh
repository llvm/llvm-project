#!/bin/bash

set -eux

_script_dir="$(dirname "$(realpath "${0}")")"
bind_point='/aocc'

image_tail=containers/AOCC_builder/aocc-rhel8:0.4-rh8.9-gcc8.5.0.sif # Code smell: parameterize?
image_path=/proj/csg_tools/${image_tail}
if test ! -f "${image_path}"
then
  image_path=/proj/csse_jenkins2/swtools/${image_tail}
fi
envs=(
  --env AOCC_DUAL_FLANG_BUILD="${AOCC_DUAL_FLANG_BUILD:-false}"
  --env BRANCH_NAME="${BRANCH_NAME:-NoSuchBranchName}"
  --env BUILD_NUMBER="${BUILD_NUMBER:-0}"
  --env CI="${CI:-false}"
  --env JOB_NAME="${JOB_NAME:-NoSuchJobName}"
)
if test -n "${BUILD_TYPE:-}"
then
  envs+=(--env BUILD_TYPE="${BUILD_TYPE}")
fi
if test -n "${CHANGE_TARGET:-}"
then
  envs+=(--env CHANGE_TARGET="${CHANGE_TARGET}")
fi
if test -n "${REL_MODE:-}"
then
  envs+=(--env REL_MODE="${REL_MODE}")
fi
if test -n "${STAGE_BUILD_NUM:-}"
then
  envs+=(--env STAGE_BUILD_NUM="${STAGE_BUILD_NUM}")
fi
if test -n "${VERSION_VAL:-}"
then
  envs+=(--env VERSION_VAL="${VERSION_VAL}")
fi
binds="/proj:/proj:ro,$(pwd):${bind_point}:rw"
if test -n "${JENKINS_REMOTE_GIT_CACHE:-}"
then
  binds+=",${JENKINS_REMOTE_GIT_CACHE}:${JENKINS_REMOTE_GIT_CACHE}:ro"
fi
apptainer exec \
          --cleanenv \
          "${envs[@]}" \
          --bind "${binds}" \
          ${image_path} \
          bash -c "cd ${bind_point} && ${_script_dir}/run.sh"
