// Test: Invalid output path (%t is a file, not a directory)
// RUN: rm -rf %t && echo "not a dir" > %t
// RUN: not clang-doc %s -output=%t/subdir 2>&1 | FileCheck --check-prefix=BAD-OUTPUT %s

// BAD-OUTPUT: clang-doc error:
// BAD-OUTPUT: {{(Not a directory|no such file or directory)}}

//
// Test: Invalid format option
// RUN: not clang-doc %s -format=badformat 2>&1 | FileCheck --check-prefix=BAD-FORMAT %s

// BAD-FORMAT: clang-doc: for the --format option: Cannot find option named 'badformat'!