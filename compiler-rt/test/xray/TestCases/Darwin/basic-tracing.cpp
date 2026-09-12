// Verify that XRay instrumentation sections are correctly emitted in Mach-O
// and that the runtime can discover them.

// RUN: %clangxx_xray -fxray-instruction-threshold=1 %s -o %t
// RUN: otool -l %t | FileCheck %s --check-prefix SECTION
// RUN: %run %t 2>&1 | FileCheck %s

// REQUIRES: target={{(arm64|x86_64)-apple-.*}}

// SECTION: sectname xray_instr_map
// SECTION-NEXT: segname __DATA

#include "xray/xray_interface.h"
#include <cstdio>

[[clang::xray_always_instrument]] int instrumented_fn() { return 42; }

[[clang::xray_never_instrument]] void handler(int32_t fid, XRayEntryType type) {
  if (type == XRayEntryType::ENTRY)
    printf("entry: %d\n", fid);
  else if (type == XRayEntryType::EXIT)
    printf("exit: %d\n", fid);
}

int main() {
  __xray_set_handler(handler);
  __xray_patch();
  int r = instrumented_fn();
  __xray_unpatch();
  printf("result: %d\n", r);
  return 0;
}

// CHECK: entry: {{[0-9]+}}
// CHECK: exit: {{[0-9]+}}
// CHECK: result: 42
