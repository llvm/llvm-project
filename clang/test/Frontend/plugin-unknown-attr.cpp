// Tests that a plugin can recover an unrecognized C++ [[...]] attribute from the
// AST. Clang does not implement these attributes, so it retains them as
// UnknownAttr (after -Wunknown-attributes); the PrintFunctionNames example
// plugin reads them back and reports the scope::name and argument text, instead
// of losing them.

// RUN: %clang_cc1 -std=c++17 -load %llvmshlibdir/PrintFunctionNames%pluginext \
// RUN:   -plugin print-fns -Wno-unknown-attributes %s 2>&1 | FileCheck %s

// REQUIRES: plugins, examples

[[vendor::transient(a, b)]] int x;
// CHECK: top-level-decl: "x"
// CHECK-NEXT: unknown-attribute: "vendor::transient(a, b)"

[[frobble]] void g();
// CHECK: top-level-decl: "g"
// CHECK-NEXT: unknown-attribute: "frobble"

// A recognized attribute is not retained as unknown, so it is not reported.
[[nodiscard]] int h();
// CHECK: top-level-decl: "h"
// CHECK-NOT: unknown-attribute
