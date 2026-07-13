//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test unwinding through a frame whose return address points to the VAPI
// glue for the Live Library Update feature.

// REQUIRES: target={{.+}}-aix{{.*}}

#include <stdlib.h>
#include <stdio.h>

// VAPI glue addresses
constexpr uintptr_t vapi_glue_addr_ext_32 = 0x8B80;
constexpr uintptr_t vapi_addr_64 = 0x8e00;
constexpr uintptr_t vapi_size_64 = 0x0200;
constexpr uintptr_t vapi_glue_addr_begin = vapi_glue_addr_ext_32;     // Start address in 32-bit
constexpr uintptr_t vapi_glue_addr_end = vapi_addr_64 + vapi_size_64; // End address in 64-bit

struct FunctionDescriptor {
  uintptr_t entry;
  uintptr_t toc;
  uintptr_t env;
};

struct Tracker {
  const char* name;
  Tracker(const char* n) : name(n) { printf("Construct %s\n", name); }
  ~Tracker() { printf("Destruct %s\n", name); }
};

int comp(const void*, const void*) {
  printf("comp()\n");
  throw 42;
}

static void do_search() {
  Tracker t("do_search-local");
  char buf[1];
  char key = 0;
  printf("calling bsearch()\n");
  // bsearch uses VAPI and sets the return address to the VAPI glue when LLU
  // is on.
  bsearch(&key, buf, 1, 1, comp);
  printf("after bsearch\n");
}

int main() {
  FunctionDescriptor* fd = reinterpret_cast<FunctionDescriptor*>(bsearch);

  if (vapi_glue_addr_begin <= fd->entry && fd->entry < vapi_glue_addr_end) {
    printf("!!! Live Library Update is enabled !!!\n");
  } else {
    printf("!!! Live Library Update is NOT enabled !!!\n");
  }

  try {
    do_search();
  } catch (...) {
    printf("caught ...\n");
  }
}
