#!/bin/bash

RUNTIME_INPUTS_DIR="${TARGET_BUILD_DIR}/Inputs"
echo "Making runtime Inputs directory: $RUNTIME_INPUTS_DIR"
mkdir -p "$RUNTIME_INPUTS_DIR"

for input_dir in $(find unittests -type d -name Inputs); do
    echo "Copying $input_dir into $RUNTIME_INPUTS_DIR"
    cp -r "${input_dir}"/* "${RUNTIME_INPUTS_DIR}/"
done
