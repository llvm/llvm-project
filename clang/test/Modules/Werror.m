// RUN: rm -rf %t
// RUN: rm -rf %t-saved
// RUN: mkdir -p %t-saved

// Initial module build (-Werror=header-guard)
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash \
// RUN:     -F %S/Inputs -fsyntax-only %s -verify -Wno-incomplete-umbrella  \
// RUN:     -Werror=header-guard
// RUN: cp %t/Module.pcm %t-saved/Module.pcm

// Building with looser -Werror options does not rebuild
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash \
// RUN:     -F %S/Inputs -fsyntax-only %s -verify -Wno-incomplete-umbrella 
// RUN: diff %t/Module.pcm %t-saved/Module.pcm

// Make the build more restricted (-Werror)
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash \
// RUN:     -F %S/Inputs -fsyntax-only %s -verify -Wno-incomplete-umbrella \
// RUN:     -Werror -Wno-incomplete-umbrella
// RUN: not diff %t/Module.pcm %t-saved/Module.pcm
// RUN: cp %t/Module.pcm %t-saved/Module.pcm

// Ensure -Werror=header-guard is less strict than -Werror
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash \
// RUN:     -F %S/Inputs -fsyntax-only %s -verify -Wno-incomplete-umbrella \
// RUN:     -Werror=header-guard -Wno-incomplete-umbrella
// RUN: diff %t/Module.pcm %t-saved/Module.pcm

// But -Werror=unused is not, because some of its diags are DefaultIgnore
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash \
// RUN:     -F %S/Inputs -fsyntax-only %s -verify -Wno-incomplete-umbrella \
// RUN:     -Werror=unused
// RUN: not diff %t/Module.pcm %t-saved/Module.pcm
// RUN: cp %t/Module.pcm %t-saved/Module.pcm

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash \
// RUN:     -F %S/Inputs -fsyntax-only %s -verify -Wno-incomplete-umbrella \
// RUN:     -Werror -Wno-incomplete-umbrella

// FIXME: when rebuilding the module, take the union of the diagnostic options
// so that we don't need to rebuild here
// RUN-DISABLED: diff %t/Module.pcm %t-saved/Module.pcm

// -Wno-everything, -Werror
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash \
// RUN:     -F %S/Inputs -fsyntax-only %s -verify -Wno-incomplete-umbrella \
// RUN:     -Wno-everything -Wall -Werror
// RUN: cp %t/Module.pcm %t-saved/Module.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash \
// RUN:     -F %S/Inputs -fsyntax-only %s -verify -Wno-incomplete-umbrella \
// RUN:     -Wall -Werror
// RUN: not diff %t/Module.pcm %t-saved/Module.pcm

// -pedantic, -Werror is not compatible with -Wall -Werror
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash \
// RUN:     -F %S/Inputs -fsyntax-only %s -verify -Wno-incomplete-umbrella \
// RUN:     -Werror -pedantic
// RUN: not diff %t/Module.pcm %t-saved/Module.pcm
// RUN: cp %t/Module.pcm %t-saved/Module.pcm

// -pedantic-errors is less strict that -pedantic, -Werror
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash \
// RUN:     -F %S/Inputs -fsyntax-only %s -verify -Wno-incomplete-umbrella \
// RUN:     -pedantic-errors
// RUN: diff %t/Module.pcm %t-saved/Module.pcm

// -Wsystem-headers does not affect non-system modules
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash \
// RUN:     -F %S/Inputs -fsyntax-only %s -verify -Wno-incomplete-umbrella \
// RUN:     -pedantic-errors -Wsystem-headers
// RUN: diff %t/Module.pcm %t-saved/Module.pcm

// expected-no-diagnostics
@import Module;
