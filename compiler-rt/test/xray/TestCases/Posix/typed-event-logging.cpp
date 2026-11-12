// RUN: %clangxx_xray %s -o %t
// RUN: env XRAY_OPTIONS=patch_premain=false:verbosity=1 %run %t 2>&1 | FileCheck %s

// REQUIRES: target={{(aarch64|x86_64)-.*linux.*}}

#include <assert.h>
#include <stdio.h>
#include "xray/xray_interface.h"

[[clang::xray_always_instrument]] void foo() {
  static constexpr char CustomLogged[] = "hello custom logging!";
  printf("before calling the custom logging...\n");
  __xray_typedevent(42, CustomLogged, sizeof(CustomLogged));
  printf("after calling the custom logging...\n");
}

static void myprinter(size_t type, const void *ptr, size_t size) {
  assert(type == 42);
  printf("%.*s\n", static_cast<int>(size), static_cast<const char*>(ptr));
}

int main() {
  // CHECK: before calling the custom logging...
  // CHECK-NEXT: after calling the custom logging...
  foo();
  __xray_set_typedevent_handler(myprinter);
  __xray_patch();
  // CHECK-NEXT: before calling the custom logging...
  // CHECK-NEXT: hello custom logging!
  // CHECK-NEXT: after calling the custom logging...
  foo();
  // CHECK-NEXT: before calling the custom logging...
  // CHECK-NEXT: after calling the custom logging...
  __xray_remove_typedevent_handler();
  foo();
}
