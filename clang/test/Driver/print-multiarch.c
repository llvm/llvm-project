// Check the output of -print-multiarch.

// RUN: %clang -print-multiarch --target=x86_64-unknown-linux-gnu \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:      | FileCheck --check-prefix=PRINT-MULTIARCH %s
// PRINT-MULTIARCH: {{^}}x86_64-linux-gnu{{$}}
