// RUN: rm -rf %t.dst %t.cache
// RUN: mkdir -p %t.dst/folder-with-modulemap %t.dst/pch-folder
// RUN: echo '#import "folder-with-modulemap/included.h"' > %t.dst/header.h
// RUN: echo 'extern int MyModuleVersion;' > %t.dst/folder-with-modulemap/MyModule.h
// RUN: echo '@import MyModule;' > %t.dst/folder-with-modulemap/included.h
// RUN: echo 'module MyModule { header "MyModule.h" }' > %t.dst/folder-with-modulemap/module.modulemap

// RUN: %clang -o %t.dst/pch-folder/header.pch -x objective-c-header -fmodules-cache-path=%t.cache -fmodules %t.dst/header.h
// RUN: not %clang -fsyntax-only -fmodules-cache-path=%t.cache -fmodules %s -include-pch %t.dst/pch-folder/header.pch 2>&1 | FileCheck %s

void test() {
  (void)MyModuleVersion; // should be found by implicit import
}

// CHECK: module 'MyModule' in AST file '{{.*MyModule.*pcm}}' (imported by AST file '[[PCH:.*header.pch]]') is not defined in any loaded module map file; maybe you need to load '[[PATH:.*folder-with-modulemap]]
// CHECK: consider adding '[[PATH]]' to the header search path
// CHECK: imported by '[[PCH]]'
