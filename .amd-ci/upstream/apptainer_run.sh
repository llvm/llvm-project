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
apptainer exec \
          --cleanenv \
          --env BUILD_NUMBER="${BUILD_NUMBER:-0}" \
          --env CI="${CI:-false}" \
          --env JOB_NAME="${JOB_NAME:-NoSuchJobName}" \
          --bind "/proj:/proj:ro,$(pwd):${bind_point}:rw" \
          ${image_path} \
          bash -c "cd ${bind_point} && ${_script_dir}/run.sh"
