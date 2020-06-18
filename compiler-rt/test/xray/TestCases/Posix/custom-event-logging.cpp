// Use the clang feature for custom xray event logging.
//
// RUN: %clangxx_xray -std=c++11 %s -o %t
// RUN: XRAY_OPTIONS="patch_premain=false verbosity=1 xray_logfile_base=custom-event-logging.xray-" %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_xray -std=c++11 -fpic -fpie %s -o %t
// RUN: XRAY_OPTIONS="patch_premain=false verbosity=1 xray_logfile_base=custom-event-logging.xray-" %run %t 2>&1 | FileCheck %s
// FIXME: Support this in non-x86_64 as well
// REQUIRES: x86_64-linux
// REQUIRES: built-in-llvm-tree
#include <cstdio>
#include "xray/xray_interface.h"

[[clang::xray_always_instrument]] void foo() {
  static constexpr char CustomLogged[] = "hello custom logging!";
  printf("before calling the custom logging...\n");
  __xray_customevent(CustomLogged, sizeof(CustomLogged));
  printf("after calling the custom logging...\n");
}

void myprinter(void* ptr, size_t size) {
  printf("%.*s\n", static_cast<int>(size), static_cast<const char*>(ptr));
}

int main() {
  foo();
  // CHECK: before calling the custom logging...
  // CHECK-NEXT: after calling the custom logging...
  printf("setting up custom event handler...\n");
  // CHECK-NEXT: setting up custom event handler...
  __xray_set_customevent_handler(myprinter);
  __xray_patch();
  // CHECK-NEXT: before calling the custom logging...
  foo();
  // CHECK-NEXT: hello custom logging!
  // CHECK-NEXT: after calling the custom logging...
  printf("removing custom event handler...\n");
  // CHECK-NEXT: removing custom event handler...
  __xray_remove_customevent_handler();
  foo();
  // CHECK-NEXT: before calling the custom logging...
  // CHECK-NEXT: after calling the custom logging...
}
