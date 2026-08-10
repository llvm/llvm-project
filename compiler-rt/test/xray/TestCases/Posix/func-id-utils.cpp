// Check that we can turn a function id to a function address, and also get the
// maximum function id for the current binary.
//
// RUN: %clangxx_xray -std=c++11 %s -o %t
// RUN: env XRAY_OPTIONS="patch_premain=false" %run %t
// RUN: %clangxx_xray -fno-xray-function-index -std=c++11 %s -o %t
// RUN: env XRAY_OPTIONS="patch_premain=false" %run %t

// UNSUPPORTED: target-is-mips64,target-is-mips64el

#include "xray/xray_interface.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iterator>
#include <set>

[[clang::xray_always_instrument]] void bar(){}

[[clang::xray_always_instrument]] void foo() {
  bar();
}

[[clang::xray_always_instrument]] int main(int argc, char *argv[]) {
  assert(__xray_max_function_id() != 0 && "we need xray instrumentation!");
  std::set<void *> must_be_instrumented = {reinterpret_cast<void *>(&foo),
                                           reinterpret_cast<void *>(&bar),
                                           reinterpret_cast<void *>(&main)};
  std::set<void *> all_instrumented;
  for (auto i = __xray_max_function_id(); i != 0; --i) {
    auto addr = __xray_function_address(i);
    all_instrumented.insert(reinterpret_cast<void *>(addr));
  }
  assert(all_instrumented.size() == __xray_max_function_id() &&
         "each function id must be assigned to a unique function");

  std::set<void *> not_instrumented;
  std::set_difference(
      must_be_instrumented.begin(), must_be_instrumented.end(),
      all_instrumented.begin(), all_instrumented.end(),
      std::inserter(not_instrumented, not_instrumented.begin()));
  assert(
      not_instrumented.empty() &&
      "we should see all explicitly instrumented functions with function ids");
  return not_instrumented.empty() ? 0 : 1;
}
