// RUN: rm -rf %t %t-unshared
// RUN: sed -e "s@INPUT_DIR@%{/S:regex_replacement}/Inputs@g" -e "s@OUT_DIR@%{/t:regex_replacement}@g" %S/Inputs/vfsoverlay.yaml > %t.yaml
// RUN: %clang_cc1 -Werror -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -ivfsoverlay %t.yaml -I %t -fsyntax-only %s
// RUN: %clang_cc1 -Werror -fmodules -fno-modules-share-filemanager -fimplicit-module-maps -fmodules-cache-path=%t-unshared -ivfsoverlay %t.yaml -I %t -fsyntax-only %s

@import not_real;

void foo(void) {
  bar();
}

// Import a submodule that is defined in actual_module2.map, which is only
// mapped in vfsoverlay2.yaml.
#ifdef IMPORT2
@import not_real.from_second_module;
// CHECK-VFS2: error: no submodule
#endif

// Override the module map (vfsoverlay2 on top)
// RUN: sed -e "s@INPUT_DIR@%{/S:regex_replacement}/Inputs@g" -e "s@OUT_DIR@%{/t:regex_replacement}@g" %S/Inputs/vfsoverlay2.yaml > %t2.yaml
// RUN: %clang_cc1 -Werror -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -ivfsoverlay %t.yaml -ivfsoverlay %t2.yaml -I %t -fsyntax-only %s
// RUN: %clang_cc1 -Werror -fmodules -fno-modules-share-filemanager -fimplicit-module-maps -fmodules-cache-path=%t-unshared -ivfsoverlay %t.yaml -ivfsoverlay %t2.yaml -I %t -fsyntax-only %s

// vfsoverlay2 not present
// RUN: not %clang_cc1 -Werror -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -ivfsoverlay %t.yaml -I %t -fsyntax-only %s -DIMPORT2 2>&1 | FileCheck -check-prefix=CHECK-VFS2 %s
// RUN: not %clang_cc1 -Werror -fmodules -fno-modules-share-filemanager -fimplicit-module-maps -fmodules-cache-path=%t-unshared -ivfsoverlay %t.yaml -I %t -fsyntax-only %s -DIMPORT2 2>&1 | FileCheck -check-prefix=CHECK-VFS2 %s

// vfsoverlay2 on the bottom
// RUN: not %clang_cc1 -Werror -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -ivfsoverlay %t2.yaml -ivfsoverlay %t.yaml -I %t -fsyntax-only %s -DIMPORT2 2>&1 | FileCheck -check-prefix=CHECK-VFS2 %s
// RUN: not %clang_cc1 -Werror -fmodules -fno-modules-share-filemanager -fimplicit-module-maps -fmodules-cache-path=%t-unshared -ivfsoverlay %t2.yaml -ivfsoverlay %t.yaml -I %t -fsyntax-only %s -DIMPORT2 2>&1 | FileCheck -check-prefix=CHECK-VFS2 %s
