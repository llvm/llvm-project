/// Invalid output path (%t is a file, not a directory)
// RUN: rm -rf %t && touch %t
// RUN: not clang-doc %s -output=%t/subdir 2>&1 | FileCheck %s
// CHECK: clang-doc error:
// CHECK: {{(Not a directory|no such file or directory)}}

/// Invalid format option
// RUN: not clang-doc %s -format=badformat 2>&1 | FileCheck %s --check-prefix=BAD-FORMAT
// BAD-FORMAT: clang-doc: for the --format option: Cannot find option named 'badformat'!
