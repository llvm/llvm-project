// RUN: rm -rf %t

// Here, we build the module without "non-modular-include-in-framework-module".
// RUN: echo '@import UseAfterFreePublic;' | \
// RUN:   %clang_cc1 -fmodules -fimplicit-module-maps \
// RUN:     -fmodules-cache-path=%t -isystem %S/Inputs/UseAfterFree/ -fsyntax-only \
// RUN:     -x objective-c -

// RUN:   %clang_cc1 -fmodules -fimplicit-module-maps \
// RUN:     -fmodules-cache-path=%t -isystem %S/Inputs/UseAfterFree/ -fsyntax-only \
// RUN:     -Wnon-modular-include-in-framework-module -Werror=non-modular-include-in-framework-module \
// RUN:     -x objective-c %s -verify
// expected-no-diagnostics

// Here, we load the module UseAfterFreePublic, it is treated as a system module,
// we ignore the inconsistency for "non-modular-include-in-framework-module".
@import UseAfterFreePublic;

// We start a thread to build the module for UseAfterFreePrivate.h. In the thread,
// we load UseAfterFreePublic and should treat it as a system module as well.
// If not, we will invalidate UseAfterFreePublic because of the inconsistency
// for "non-modular-include-in-framework-module", and have a use-after-free error
// of the FileEntry.
#import <UseAfterFreePrivate.h>
