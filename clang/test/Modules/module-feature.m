// RUN: rm -rf %t %t.nohash

// Each set of features gets its own cache.
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -fmodule-feature f1 -fmodule-feature f2 -F %S/Inputs %s -verify -Rmodule-build
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -fmodule-feature f2 -F %S/Inputs %s -verify -Rmodule-build
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -fmodule-feature f2 -fmodule-feature f1 -F %S/Inputs %s -Rmodule-build 2>&1 | FileCheck %s -allow-empty -check-prefix=ALREADY_BUILT
// ALREADY_BUILT-NOT: building module

// Errors if we try to force the load.
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t.nohash -fimplicit-module-maps -fdisable-module-hash -fmodule-feature f1 -fmodule-feature f2 -F %S/Inputs %s -verify -Rmodule-build
// RUN: not %clang_cc1 -fmodules -fmodules-cache-path=%t.nohash -fimplicit-module-maps -fdisable-module-hash -fmodule-feature f2 -F %S/Inputs %s 2>&1 | FileCheck %s -check-prefix=DIFFERS
// DIFFERS: error: module features differs

@import Module; // expected-remark {{building module 'Module'}} expected-remark {{finished}}
