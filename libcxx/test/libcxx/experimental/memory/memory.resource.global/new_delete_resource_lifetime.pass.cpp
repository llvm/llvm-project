//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Aligned allocation is required by std::experimental::pmr, but it was not provided
// before macosx10.13 and as a result we get linker errors when deploying to older than
// macosx10.13.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}}

// <experimental/memory_resource>

// memory_resource * new_delete_resource()

// The lifetime of the value returned by 'new_delete_resource()' should
// never end, even very late into program termination. This test constructs
// attempts to use 'new_delete_resource()' very late in program termination
// to detect lifetime issues.

#include <experimental/memory_resource>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

namespace ex = std::experimental::pmr;

struct POSType {
  ex::memory_resource* res = nullptr;
  void* ptr = nullptr;
  int n = 0;
  POSType() {res = ex::new_delete_resource(); ptr = res->allocate(42); n = 42; }
  POSType(ex::memory_resource* r, void* p, int s) : res(r), ptr(p), n(s) {}
  ~POSType() { if (ptr) res->deallocate(ptr, n); }
};

void swap(POSType & L, POSType & R) {
    std::swap(L.res, R.res);
    std::swap(L.ptr, R.ptr);
    std::swap(L.n, R.n);
}

POSType constructed_before_resources;

// Constructs resources
ex::memory_resource* resource = ex::new_delete_resource();

POSType constructed_after_resources(resource, resource->allocate(1024), 1024);

int main(int, char**)
{
    swap(constructed_after_resources, constructed_before_resources);

  return 0;
}
