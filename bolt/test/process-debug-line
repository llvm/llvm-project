#!/bin/sh

# Script to process llvm-dwarfdump --debug-line output and create a normalized table
# Usage: process-debug-line.sh <debug-line.txt>
#
# Output format: CU_FILE LINE COLUMN FILE_NAME [additional_info]
# This strips addresses to make rows unique and adds context about which CU and file each line belongs to

if [ $# -ne 1 ]; then
    echo "Usage: $0 <debug-line.txt>" >&2
    exit 1
fi

debug_line_file="$1"

if [ ! -f "$debug_line_file" ]; then
    echo "Error: File '$debug_line_file' not found" >&2
    exit 1
fi

awk '
BEGIN {
    cu_count = 0
    current_cu_file = ""
    # Initialize file names array
    for (i = 0; i < 100; i++) {
        current_file_names[i] = ""
    }
}

# Track debug_line sections (new CU)
/^debug_line\[/ {
    cu_count++
    current_cu_file = ""
    # Clear file names array for new CU
    for (i = 0; i < 100; i++) {
        current_file_names[i] = ""
    }
    next
}

# Capture file names and their indices
/^file_names\[.*\]:/ {
    # Extract file index using simple string operations
    line_copy = $0
    gsub(/file_names\[/, "", line_copy)
    gsub(/\]:.*/, "", line_copy)
    gsub(/[ \t]/, "", line_copy)
    file_index = line_copy

    getline  # Read the next line which contains the actual filename
    # Extract filename from name: "filename" format
    if (match($0, /name:[ \t]*"/)) {
        filename = $0
        gsub(/.*name:[ \t]*"/, "", filename)
        gsub(/".*/, "", filename)
        current_file_names[file_index] = filename

        # Extract basename for main CU file (first .c/.cpp/.cc file we see)
        if (current_cu_file == "" && match(filename, /\.(c|cpp|cc)$/)) {
            cu_filename = filename
            gsub(/.*\//, "", cu_filename)
            current_cu_file = cu_filename
        }
    }
    next
}

# Process line table entries
/^0x[0-9a-f]+/ {
    # Parse the line entry: Address Line Column File ISA Discriminator OpIndex Flags
    if (NF >= 4) {
        line = $2
        column = $3
        file_index = $4

        # Get the filename for this file index
        filename = current_file_names[file_index]
        if (filename == "") {
            filename = "UNKNOWN_FILE_" file_index
        } else {
            # Extract just the basename
            basename = filename
            gsub(/.*\//, "", basename)
            filename = basename
        }

        # Build additional info (flags, etc.)
        additional_info = ""
        for (i = 8; i <= NF; i++) {
            if (additional_info != "") {
                additional_info = additional_info " "
            }
            additional_info = additional_info $i
        }

        # Output normalized row: CU_FILE LINE COLUMN FILE_NAME [additional_info]
        printf "%s %s %s %s", current_cu_file, line, column, filename
        if (additional_info != "") {
            printf " %s", additional_info
        }
        printf "\n"
    }
}
' "$debug_line_file"
