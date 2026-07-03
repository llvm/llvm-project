/// Invalid output path (%t is a file, not a directory).
// RUN: rm -rf %t && touch %t
// RUN: not clang-doc %s -output=%t/subdir 2>&1 | FileCheck %s --check-prefix=OUTPUT-FAIL
// OUTPUT-FAIL: clang-doc error:
// OUTPUT-FAIL-SAME: failed to create directory.

/// Invalid format option.
// RUN: rm -rf %t && mkdir %t && touch %t/file
// RUN: not clang-doc %s --output=%t/file -format=badformat 2>&1 | FileCheck %s --check-prefix=BAD-FORMAT
// BAD-FORMAT: clang-doc: for the --format option: Cannot find option named 'badformat'!

/// Missing HTML asset directory (warning only).
// RUN: clang-doc %s -format=html -asset=%t/nonexistent-assets 2>&1 | FileCheck %s --check-prefix=ASSET-WARN
// ASSET-WARN: Asset path supply is not a directory

/// Mapping failure (with --ignore-map-errors=false).
// RUN: not clang-doc %t/nonexistent.cpp -ignore-map-errors=false 2>&1 | FileCheck %s --check-prefix=MAP-FAIL
// MAP-FAIL: clang-doc error: Failed to run action

/// Mapping failure (with --ignore-map-errors=true).
// RUN: clang-doc %t/nonexistent.cpp 2>&1 | FileCheck %s --check-prefix=MAP-WARN
// MAP-WARN: Error mapping decls in files. Clang-doc will ignore these files and continue

///Invalid executor type
// RUN: not clang-doc --executor=invalid %s 2>&1 | FileCheck %s --check-prefix=EXECUTOR-FAIL
// EXECUTOR-FAIL: clang-doc error:
// EXECUTOR-FAIL: Executor "invalid" is not registered

///TODO: Add tests for failures in generateDocs() and in createResources().
