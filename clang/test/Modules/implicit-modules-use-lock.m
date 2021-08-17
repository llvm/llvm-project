// Confirm -fimplicit-modules-use-lock and -fno-implicit-modules-use-lock control
// whether building a module triggers -Rmodule-lock, indirectly checking whether
// a lock manager is being used.
//
// RUN: rm -rf %t.cache
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps \
// RUN:   -fimplicit-modules-use-lock -Rmodule-lock -Rmodule-build \
// RUN:   -fmodules-cache-path=%t.cache -I%S/Inputs/system-out-of-date \
// RUN:   -fsyntax-only %s -Wnon-modular-include-in-framework-module \
// RUN:   -Werror=non-modular-include-in-framework-module \
// RUN:   -verify=locks,build
//
// RUN: rm -rf %t.cache
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps \
// RUN:   -fno-implicit-modules-use-lock -Rmodule-lock -Rmodule-build \
// RUN:   -fmodules-cache-path=%t.cache -I%S/Inputs/system-out-of-date \
// RUN:   -fsyntax-only %s -Wnon-modular-include-in-framework-module \
// RUN:   -Werror=non-modular-include-in-framework-module \
// RUN:   -verify=build

@import X; // locks-remark-re {{locking '{{.*}}.pcm' to build module 'X'}} \
           // build-remark {{building module 'X'}}                         \
           // build-remark {{finished building module 'X'}}
