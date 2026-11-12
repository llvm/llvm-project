// RUN: rm -rf %t

// Including a header from the imported module
// RUN: echo '@import FromImportedModuleOK;' | \
// RUN:   %clang_cc1 -Wnon-modular-include-in-framework-module -fmodules -fimplicit-module-maps \
// RUN:     -fmodules-cache-path=%t -F %S/Inputs/require-modular-includes \
// RUN:     -Werror -fsyntax-only -x objective-c -

// Including a non-modular header
// RUN: echo '@import FromImportedModuleFail;' | \
// RUN:   %clang_cc1 -Wnon-modular-include-in-framework-module -fmodules -fimplicit-module-maps \
// RUN:     -fmodules-cache-path=%t -F %S/Inputs/require-modular-includes \
// RUN:     -I %S/Inputs/require-modular-includes \
// RUN:     -fsyntax-only -x objective-c - 2>&1 | FileCheck %s

// Including a header from a subframework
// RUN: echo '@import FromSubframework;' | \
// RUN:   %clang_cc1 -Wnon-modular-include-in-framework-module -fmodules -fimplicit-module-maps \
// RUN:     -fmodules-cache-path=%t -F %S/Inputs/require-modular-includes \
// RUN:     -Werror -fsyntax-only -x objective-c -

// Including a header from a subframework (fail)
// RUN: echo '@import FromNonModularSubframework;' | \
// RUN:   %clang_cc1 -Wnon-modular-include-in-framework-module -fmodules -fimplicit-module-maps \
// RUN:     -fmodules-cache-path=%t -F %S/Inputs/require-modular-includes \
// RUN:     -I %S/Inputs/require-modular-includes \
// RUN:     -fsyntax-only -x objective-c - 2>&1 | FileCheck %s

// Including a non-modular header from a submodule
// RUN: echo '@import FromImportedSubModule;' | \
// RUN:   %clang_cc1 -Wnon-modular-include-in-framework-module -fmodules -fimplicit-module-maps \
// RUN:     -fmodules-cache-path=%t -F %S/Inputs/require-modular-includes \
// RUN:     -I %S/Inputs/require-modular-includes \
// RUN:     -fsyntax-only -x objective-c - 2>&1 | FileCheck %s

// Including a non-modular header (directly) with -fmodule-name set
// RUN: echo '#include "NotInModule.h"' | \
// RUN:   %clang_cc1 -Wnon-modular-include-in-framework-module -fmodules -fimplicit-module-maps \
// RUN:     -fmodules-cache-path=%t -I %S/Inputs/require-modular-includes \
// RUN:     -Werror -fmodule-name=A -fsyntax-only -x objective-c -

// Including an excluded header
// RUN: echo '@import IncludeExcluded;' | \
// RUN:   %clang_cc1 -Wnon-modular-include-in-framework-module -fmodules -fimplicit-module-maps \
// RUN:     -fmodules-cache-path=%t -F %S/Inputs/require-modular-includes \
// RUN:     -Werror -fsyntax-only -x objective-c -

// Including a header from another module
// RUN: echo '@import FromAnotherModule;' | \
// RUN:   %clang_cc1 -Wnon-modular-include-in-framework-module -fmodules -fimplicit-module-maps \
// RUN:     -fmodules-cache-path=%t -F %S/Inputs/require-modular-includes \
// RUN:     -I %S/Inputs/require-modular-includes \
// RUN:     -Werror -fsyntax-only -x objective-c -

// Including an excluded header from another module
// RUN: echo '@import ExcludedFromAnotherModule;' | \
// RUN:   %clang_cc1 -Wnon-modular-include-in-framework-module -fmodules -fimplicit-module-maps \
// RUN:     -fmodules-cache-path=%t -F %S/Inputs/require-modular-includes \
// RUN:     -I %S/Inputs/require-modular-includes \
// RUN:     -Werror -fsyntax-only -x objective-c -

// Including a header from an umbrella directory
// RUN: echo '@import FromUmbrella;' | \
// RUN:   %clang_cc1 -Wnon-modular-include-in-framework-module -fmodules -fimplicit-module-maps \
// RUN:     -fmodules-cache-path=%t -F %S/Inputs/require-modular-includes \
// RUN:     -I %S/Inputs/require-modular-includes \
// RUN:     -Werror -fsyntax-only -x objective-c -

// A includes B includes non-modular C
// RUN: echo '@import A;' | \
// RUN:   %clang_cc1 -Wnon-modular-include-in-framework-module -fmodules -fimplicit-module-maps \
// RUN:     -fmodules-cache-path=%t -F %S/Inputs/require-modular-includes \
// RUN:     -I %S/Inputs/require-modular-includes \
// RUN:     -fsyntax-only -x objective-c - 2>&1 | FileCheck %s

// Non-framework module (pass)
// RUN: echo '@import NotFramework;' | \
// RUN:   %clang_cc1 -Wnon-modular-include-in-framework-module -fmodules -fimplicit-module-maps \
// RUN:     -fmodules-cache-path=%t -I %S/Inputs/require-modular-includes \
// RUN:     -Werror -fsyntax-only -x objective-c -

// CHECK: include of non-modular header
