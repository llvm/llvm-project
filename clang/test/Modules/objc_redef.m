@import redeclarations_left;
@import weird_objc;
@import objc_redef_indirect;

int test(id x) {
  return x->wibble;
}

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -fmodules-cache-path=%t -emit-module -fmodule-name=redeclarations_left %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -fmodules-cache-path=%t -emit-module -fmodule-name=weird_objc %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify

// Try explicit too.

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -emit-module -fmodule-name=redeclarations_left %S/Inputs/module.map -o %t/redeclarations_left.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -emit-module -fmodule-name=weird_objc %S/Inputs/module.map -o %t/weird_objc.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -emit-module -fmodule-file=%t/weird_objc.pcm -fmodule-name=objc_redef_indirect %S/Inputs/module.map -o %t/objc_redef_indirect.pcm -I %S/Inputs
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodule-file=%t/redeclarations_left.pcm -fmodule-file=%t/weird_objc.pcm -fmodule-file=%t/objc_redef_indirect.pcm -I %S/Inputs %s -verify
// expected-no-diagnostics

