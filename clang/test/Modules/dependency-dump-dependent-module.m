// When a module depends on another, check that we dump the dependency header
// files for both.

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -module-dependency-dir %t/vfs -F %S/Inputs -I %S/Inputs -verify %s
// expected-no-diagnostics

// RUN: FileCheck %s -check-prefix=VFS < %t/vfs/vfs.yaml
// VFS: 'name': "AlsoDependsOnModule.h"
// VFS: 'name': "SubFramework.h"
// VFS: 'name': "Treasure.h"
// VFS: 'name': "Module.h"
// VFS: 'name': "Sub.h"
// VFS: 'name': "Sub2.h"

@import AlsoDependsOnModule;
