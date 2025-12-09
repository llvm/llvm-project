// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=json --executor=standalone %s
// RUN: FileCheck %s < %t/json/foo/tools/index.json --check-prefix=CHECK-FOO
// RUN: FileCheck %s < %t/json/bar/tools/index.json --check-prefix=CHECK-BAR

namespace foo {
  namespace tools {
    class FooTools {};
  } // namespace tools
} // namespace foo

namespace bar {
  namespace tools {
    class BarTools {};
  } // namespace tools
} // namespace bar

// CHECK-FOO: "Name": "tools"

// CHECK-BAR: "Name": "tools"
