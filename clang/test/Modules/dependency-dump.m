// Check that we can dump all of the headers a module depends on, and a VFS map
// for the same.

// REQUIRES: shell

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -module-dependency-dir %t/vfs -F %S/Inputs -I %S/Inputs -verify %s
// expected-no-diagnostics

// RUN: FileCheck %s -check-prefix=VFS -input-file %t/vfs/vfs.yaml
// VFS: 'name': "SubFramework.h"
// VFS: 'name': "Treasure.h"
// VFS: 'name': "Module.h"
// VFS: 'name': "Sub.h"
// VFS: 'name': "Sub2.h"

@import Module;
