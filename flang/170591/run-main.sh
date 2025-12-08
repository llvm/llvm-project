#!/bin/bash

# Get the directory of the script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
REPRO_DIR="$SCRIPT_DIR"

# Define the flang compiler path
FLANG_COMPILER="/home/eepshteyn/compilers/flang-upstream/bin/flang"

# Compile the Fortran code with -O3 optimization
echo "Compiling repro-main.f90 with -O3..."
"$FLANG_COMPILER" -O3 -S "$REPRO_DIR/repro-main.f90" -o "$REPRO_DIR/repro-main.s"

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Assembly generated in repro-main.s"
