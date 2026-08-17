// End-to-end test: compile with XRay, run with handler, verify the full
// instrumentation pipeline works (section discovery, patching, trampoline,
// handler callback, unpatching).

// RUN: %clangxx_xray -fxray-instruction-threshold=1 %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %llvm_xray extract %t | FileCheck %s --check-prefix MAP

// REQUIRES: target={{(arm64|x86_64)-apple-.*}}

// Verify the instrumentation map can be read by llvm-xray
// MAP: - { id: {{[0-9]+}}, address: 0x{{[0-9a-fA-F]+}}, function: 0x{{[0-9a-fA-F]+}}

#include "xray/xray_interface.h"
#include <cstdio>

static int entries = 0;
static int exits = 0;

[[clang::xray_never_instrument]] void handler(int32_t fid, XRayEntryType type) {
  if (type == XRayEntryType::ENTRY)
    ++entries;
  else if (type == XRayEntryType::EXIT)
    ++exits;
}

[[clang::xray_always_instrument]] int target_fn(int x) { return x * 2; }

int main() {
  __xray_set_handler(handler);
  __xray_patch();
  int sum = 0;
  for (int i = 0; i < 5; ++i)
    sum += target_fn(i);
  __xray_unpatch();
  // 5 calls to target_fn = 5 entries + 5 exits
  printf("sum=%d entries=%d exits=%d\n", sum, entries, exits);
  return (entries == 5 && exits == 5) ? 0 : 1;
}

// CHECK: sum=20 entries=5 exits=5
