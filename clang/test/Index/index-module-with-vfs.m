@import ModuleNeedsVFS;

void foo() {
  module_needs_vfs();
  base_module_needs_vfs();
}

// RUN: rm -rf %t.cache
// RUN: sed -e "s@INPUT_DIR@%{/S:regex_replacement}/Inputs@g" -e "s@OUT_DIR@%{/t:regex_replacement}@g" %S/Inputs/vfsoverlay.yaml > %t.yaml
// RUN: c-index-test -index-file %s -fmodules-cache-path=%t.cache -fmodules -F %t -I %t \
// RUN:              -ivfsoverlay %t.yaml -Xclang -fdisable-module-hash | FileCheck %s

// CHECK: [importedASTFile]: {{.*}}ModuleNeedsVFS.pcm | loc: 1:1 | name: "ModuleNeedsVFS" | isImplicit: 0
// CHECK: [indexEntityReference]: kind: function | name: module_needs_vfs
// CHECK: [indexEntityReference]: kind: function | name: base_module_needs_vfs

// RUN: c-index-test -index-tu %t.cache/ModuleNeedsVFS.pcm | FileCheck %s -check-prefix=CHECK-MOD

// CHECK-MOD: [ppIncludedFile]: {{.*}}module_needs_vfs.h 
// CHECK-MOD: [importedASTFile]: {{.*}}BaseModuleNeedsVFS.pcm
// CHECK-MOD: [indexEntityReference]: kind: function | name: base_module_needs_vfs

// RUN: c-index-test -index-tu %t.cache/BaseModuleNeedsVFS.pcm | FileCheck %s -check-prefix=CHECK-MOD2

// CHECK-MOD2: [ppIncludedFile]: {{.*}}base_module_needs_vfs.h
