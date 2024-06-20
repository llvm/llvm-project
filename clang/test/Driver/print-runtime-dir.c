// Per-target directory layout
// RUN: %clang -print-runtime-dir --target=x86_64-pc-windows-msvc \
// RUN:   -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   | FileCheck --check-prefix=PRINT-RUNTIME-DIR-PER-TARGET -DFILE=%S/Inputs/resource_dir_with_per_target_subdir  %s
// PRINT-RUNTIME-DIR-PER-TARGET: [[FILE]]{{/|\\}}lib{{/|\\}}x86_64-pc-windows-msvc
