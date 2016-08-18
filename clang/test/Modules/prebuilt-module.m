// RUN: rm -rf %t
//
// RUN: %clang_cc1 -fmodules -x objective-c -I %S/Inputs/prebuilt-module -triple %itanium_abi_triple -emit-module %S/Inputs/prebuilt-module/module.modulemap -fmodule-name=prebuilt -o %t/prebuilt.pcm
// RUN: %clang_cc1 -fmodules -triple %itanium_abi_triple -fprebuilt-module-path=%t/ -fdisable-module-hash %s -verify

// expected-no-diagnostics
@import prebuilt;
int test() {
  return a;
}
