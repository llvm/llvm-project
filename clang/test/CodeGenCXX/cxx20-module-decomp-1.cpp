// RUN: %clang_cc1 -std=c++20 %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s

export module Foo;

// CHECK-DAG: @_ZW3Foo3ary =
int ary[2];

// CHECK-DAG: @_ZW3FooDC1a1bE =
export auto &[a, b] = ary;

namespace N {
// CHECK-DAG: @_ZN1NW3FooDC1a1bEE =
export auto &[a, b] = ary;
// CHECK-DAG: @_ZN1NW3FooDC1c1dEE =
auto &[c, d] = ary;
// FIXME: We mangle the module name here, as we give this ModuleInternalLinkage
// That's no longer needed.
// CHECK DAG: @_ZN1MDC1e1fEE =
static auto &__attribute__((used))[e, f] = ary;
} // namespace N
