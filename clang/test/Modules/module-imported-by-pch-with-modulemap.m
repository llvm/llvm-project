// RUN: rm -rf %t.dst %t.cache
// RUN: mkdir -p %t.dst/folder-with-modulemap %t.dst/pch-folder
// RUN: echo '#import "folder-with-modulemap/included.h"' > %t.dst/header.h
// RUN: echo 'extern int MyModuleVersion;' > %t.dst/folder-with-modulemap/MyModule.h
// RUN: echo '@import MyModule;' > %t.dst/folder-with-modulemap/included.h
// RUN: echo 'module MyModule { header "MyModule.h" }' > %t.dst/folder-with-modulemap/MyModule.modulemap

// RUN: %clang_cc1 -emit-pch -o %t.dst/pch-folder/header.pch -fmodule-map-file=%t.dst/folder-with-modulemap/MyModule.modulemap -x objective-c-header -fmodules-cache-path=%t.cache -fmodules -fimplicit-module-maps %t.dst/header.h
// RUN: %clang_cc1 -fsyntax-only -fmodule-map-file=%t.dst/folder-with-modulemap/MyModule.modulemap -fmodules-cache-path=%t.cache -fmodules -fimplicit-module-maps %s -include-pch %t.dst/pch-folder/header.pch -verify

// expected-no-diagnostics

void test(void) {
  (void)MyModuleVersion; // should be found by implicit import
}

