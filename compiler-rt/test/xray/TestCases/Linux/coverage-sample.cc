// Check that we can patch and unpatch specific function ids.
//
// RUN: %clangxx_xray -std=c++11 %s -o %t
// RUN: XRAY_OPTIONS="patch_premain=false xray_naive_log=false" %run %t | FileCheck %s

#include "xray/xray_interface.h"

#include <set>
#include <cstdio>

std::set<int32_t> function_ids;

[[clang::xray_never_instrument]] void coverage_handler(int32_t fid,
                                                       XRayEntryType) {
  thread_local bool patching = false;
  if (patching) return;
  patching = true;
  function_ids.insert(fid);
  __xray_unpatch_function(fid);
  patching = false;
}

[[clang::xray_always_instrument]] void baz() {
  // do nothing!
}

[[clang::xray_always_instrument]] void bar() {
  baz();
}

[[clang::xray_always_instrument]] void foo() {
  bar();
}

[[clang::xray_always_instrument]] int main(int argc, char *argv[]) {
  __xray_set_handler(coverage_handler);
  __xray_patch();
  foo();
  __xray_unpatch();

  // print out the function_ids.
  printf("first pass.\n");
  for (const auto id : function_ids)
    printf("patched: %d\n", id);

  // CHECK-LABEL: first pass.
  // CHECK-DAG: patched: [[F1:.*]]
  // CHECK-DAG: patched: [[F2:.*]]
  // CHECK-DAG: patched: [[F3:.*]]

  // make a copy of the function_ids, then patch them later.
  auto called_fns = function_ids;

  // clear the function_ids.
  function_ids.clear();

  // patch the functions we've called before.
  for (const auto id : called_fns)
    __xray_patch_function(id);

  // then call them again.
  foo();
  __xray_unpatch();

  // confirm that we've seen the same functions again.
  printf("second pass.\n");
  for (const auto id : function_ids)
    printf("patched: %d\n", id);
  // CHECK-LABEL: second pass.
  // CHECK-DAG: patched: [[F1]]
  // CHECK-DAG: patched: [[F2]]
  // CHECK-DAG: patched: [[F3]]

  // Now we want to make sure that if we unpatch one, that we're only going to
  // see two calls of the coverage_handler.
  function_ids.clear();
  __xray_patch();
  __xray_unpatch_function(1);
  foo();
  __xray_unpatch();

  // confirm that we don't see function id one called anymore.
  printf("missing 1.\n");
  for (const auto id : function_ids)
    printf("patched: %d\n", id);
  // CHECK-LABEL: missing 1.
  // CHECK-NOT: patched: 1
}
