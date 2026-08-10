// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c -fmodules %S/Inputs/prebuilt-implicit-module/module.modulemap -emit-module -fmodule-name=module_a -fmodules-cache-path=%t
// RUN: find %t -name "module_a*.pcm" | grep module_a

// Check we use a prebuilt module when available, and do not build an implicit module.
// RUN: rm -rf %t1
// RUN: mkdir -p %t1
// RUN: %clang_cc1 -x objective-c %s -I%S/Inputs/prebuilt-implicit-module -fmodules -fmodule-map-file=%S/Inputs/prebuilt-implicit-module/module.modulemap -fprebuilt-implicit-modules -fprebuilt-module-path=%t -fmodules-cache-path=%t1
// RUN: find %t1 -name "module_a*.pcm" | not grep module_a

// Check a module cache path is not required when all modules resolve to
// prebuilt implicit modules.
// RUN: rm -rf %t1
// RUN: mkdir -p %t1
// RUN: %clang_cc1 -x objective-c %s -I%S/Inputs/prebuilt-implicit-module -fmodules -fmodule-map-file=%S/Inputs/prebuilt-implicit-module/module.modulemap -fprebuilt-implicit-modules -fprebuilt-module-path=%t

// Check that we correctly fall back to implicit modules if the prebuilt implicit module is not found.
// RUN: rm -rf %t1
// RUN: mkdir -p %t1
// RUN: %clang_cc1 -x objective-c %s -I%S/Inputs/prebuilt-implicit-module -fmodules -fmodule-map-file=%S/Inputs/prebuilt-implicit-module/module.modulemap -fprebuilt-implicit-modules -fprebuilt-module-path=%t -fmodules-cache-path=%t1 -fno-signed-char
// RUN: find %t1 -name "module_a*.pcm" | grep module_a

// Check that non-implicit prebuilt modules are always preferred to prebuilt implicit modules.
// RUN: rm -rf %t2
// RUN: mkdir -p %t2
// RUN: %clang_cc1 -x objective-c -fmodules %S/Inputs/prebuilt-implicit-module/module.modulemap -emit-module -fmodule-name=module_a -fmodules-cache-path=%t
// RUN: %clang_cc1 -x objective-c -fmodules %S/Inputs/prebuilt-implicit-module/module.modulemap -emit-module -fmodule-name=module_a -o %t/module_a.pcm -fno-signed-char
// RUN: not %clang_cc1 -x objective-c %s -I%S/Inputs/prebuilt-implicit-module -fmodules -fmodule-map-file=%S/Inputs/prebuilt-implicit-module/module.modulemap -fprebuilt-implicit-modules -fprebuilt-module-path=%t -fmodules-cache-path=%t2
// RUN: find %t2 -name "module_a*.pcm" | not grep module_a

// expected-no-diagnostics
@import module_a;
int test(void) {
  return a;
}
