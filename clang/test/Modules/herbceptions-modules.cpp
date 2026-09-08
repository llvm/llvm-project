// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -std=c++26 -fherbceptions -fmodules -fmodule-map-file=%S/Inputs/herbceptions.modulemap -fmodules-cache-path=%t -I %S/Inputs -verify %s
// RUN: %clang_cc1 -std=c++26 -fherbceptions -fmodules -fmodule-map-file=%S/Inputs/herbceptions.modulemap -fmodules-cache-path=%t -I %S/Inputs -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
//
// Herbception declarations and inline definitions imported from C++ modules:
// exercises deserialization of throws/fails function definitions (including
// the legacy-conversion field placement relative to the lazily-loaded body),
// cross-module references, and discriminant routing at the import site.

// expected-no-diagnostics

#include "herbceptions-b.h"

int use_throws_io() {
  try {
    throws_io();
  } catch throws(std::error e) {
    return static_cast<int>(e.code());
  }
  return 0;
}

int use_wrap_fail(int x) {
  auto e = catch return_failure(wrap_fail(x));
  return e.failed ? e.error : e.value;
}

// The imported 'throws' function keeps its attribute and {payload, i1}
// return shape; the imported inline definition is emitted (linkonce_odr).
// CHECK-DAG: define {{.*}}linkonce_odr { { ptr, i64 }, i1 } @_Z9throws_iov()
// CHECK-DAG: define {{.*}}@_Z13use_wrap_faili(
// CHECK-DAG: attributes #[[ATTR:[0-9]+]] = { {{.*}}throws{{.*}} }
