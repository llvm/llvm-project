// Verify that custom XRay events work on macOS.

// RUN: %clangxx_xray -fxray-instruction-threshold=1 %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// REQUIRES: target={{(arm64|x86_64)-apple-.*}}

#include "xray/xray_interface.h"
#include <cstdio>
#include <cstring>

static int custom_event_count = 0;
static char last_event[64] = {};

[[clang::xray_never_instrument]] void handler(int32_t fid, XRayEntryType type) {
}

[[clang::xray_never_instrument]] void custom_handler(void *data, size_t size) {
  ++custom_event_count;
  if (size < sizeof(last_event)) {
    memcpy(last_event, data, size);
    last_event[size] = '\0';
  }
}

[[clang::xray_always_instrument]] void emit_event() {
  static const char msg[] = "hello-xray";
  __xray_customevent(msg, sizeof(msg) - 1);
}

int main() {
  __xray_set_handler(handler);
  __xray_set_customevent_handler(custom_handler);
  __xray_patch();
  emit_event();
  __xray_unpatch();
  printf("events=%d\n", custom_event_count);
  return 0;
}

// CHECK: events=1
