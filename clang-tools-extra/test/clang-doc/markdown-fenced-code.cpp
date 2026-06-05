// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=md --doxygen --output=%t --executor=standalone %s 2>&1 || true
// RUN: FileCheck %s < %t/GlobalNamespace/index.md

/// A function with a fenced code block in its documentation.
/// Example usage:
/// ```cpp
/// int x = documented();
/// ```
int documented();

// CHECK: ### documented
// CHECK: ```cpp
// CHECK: int x = documented();
// CHECK: ```