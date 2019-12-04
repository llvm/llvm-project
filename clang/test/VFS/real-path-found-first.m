// This test is for cases where we lookup a file by its 'real' path before we
// use its VFS-mapped path. If we accidentally use the real path in header
// search, we will not find a module for the headers.  To test that we
// intentionally rebuild modules, since the precompiled module file refers to
// the dependency files by real path.

// RUN: rm -rf %t %t-cache %t.pch
// RUN: mkdir -p %t/SomeFramework.framework/Modules
// RUN: cat %S/Inputs/some_frame_module.map > %t/SomeFramework.framework/Modules/module.modulemap
// RUN: sed -e "s@INPUT_DIR@%{/S:regex_replacement}/Inputs@g" -e "s@OUT_DIR@%{/t:regex_replacement}@g" %S/Inputs/vfsoverlay.yaml > %t.yaml

// Build
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t-cache -F %t \
// RUN:     -ivfsoverlay %t.yaml -fsyntax-only %s -verify -Wauto-import \
// RUN:     -Werror=non-modular-include-in-framework-module

// Rebuild
// RUN: echo ' ' >> %t/SomeFramework.framework/Modules/module.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t-cache -F %t \
// RUN:     -ivfsoverlay %t.yaml -fsyntax-only %s -verify -Wauto-import \
// RUN:     -Werror=non-modular-include-in-framework-module

// Load from PCH
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t-cache -F %t \
// RUN:     -ivfsoverlay %t.yaml -emit-pch  %s -o %t.pch \
// RUN:     -Werror=non-modular-include-in-framework-module \
// RUN:     -fmodules-ignore-macro=WITH_PREFIX
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t-cache -F %t \
// RUN:     -ivfsoverlay %t.yaml -include-pch %t.pch -fsyntax-only  %s \
// RUN:     -Werror=non-modular-include-in-framework-module -DWITH_PREFIX \
// RUN:     -fmodules-ignore-macro=WITH_PREFIX

// While indexing
// RUN: c-index-test -index-file %s -fmodules -fimplicit-module-maps -fmodules-cache-path=%t-cache -F %t \
// RUN:     -ivfsoverlay %t.yaml -fsyntax-only -Wauto-import \
// RUN:     -Werror=non-modular-include-in-framework-module | FileCheck %s
// RUN: echo ' ' >> %t/SomeFramework.framework/Modules/module.modulemap
// RUN: c-index-test -index-file %s -fmodules -fimplicit-module-maps -fmodules-cache-path=%t-cache -F %t \
// RUN:     -ivfsoverlay %t.yaml -fsyntax-only -Wauto-import \
// RUN:     -Werror=non-modular-include-in-framework-module | FileCheck %s
// CHECK: warning: treating
// CHECK-NOT: error

// With a VFS-mapped module map file
// RUN: mv %t/SomeFramework.framework/Modules/module.modulemap %t/hide_module.map
// RUN: echo "{ 'version': 0, 'roots': [ { " > %t2.yaml
// RUN: echo "'name': '%t/SomeFramework.framework/Modules/module.modulemap'," >> %t2.yaml
// RUN: echo "'type': 'file', 'external-contents': '%t/hide_module.map' } ] }" >> %t2.yaml

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t-cache -F %t \
// RUN:     -ivfsoverlay %t.yaml -ivfsoverlay %t2.yaml -fsyntax-only %s -verify \
// RUN:     -Wauto-import -Werror=non-modular-include-in-framework-module
// RUN: echo ' ' >> %t/hide_module.map
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t-cache -F %t \
// RUN:     -ivfsoverlay %t.yaml -ivfsoverlay %t2.yaml -fsyntax-only %s -verify \
// RUN:     -Wauto-import -Werror=non-modular-include-in-framework-module

// Within a module build
// RUN: echo '@import import_some_frame;' | \
// RUN:   %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t-cache -F %t \
// RUN:      -ivfsoverlay %t.yaml -ivfsoverlay %t2.yaml -fsyntax-only - \
// RUN:      -Werror=non-modular-include-in-framework-module -x objective-c -I %t
// RUN: echo ' ' >> %t/hide_module.map
// RUN: echo '@import import_some_frame;' | \
// RUN:   %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t-cache -F %t \
// RUN:      -ivfsoverlay %t.yaml -ivfsoverlay %t2.yaml -fsyntax-only - \
// RUN:      -Werror=non-modular-include-in-framework-module -x objective-c -I %t

#ifndef WITH_PREFIX
#import <SomeFramework/public_header.h> // expected-warning{{treating}}
#import <SomeFramework/public_header2.h> // expected-warning{{treating}}
#import <SomeFramework/public_header3.h> // expected-warning{{treating}}
@import SomeFramework.public_header2;
#endif
