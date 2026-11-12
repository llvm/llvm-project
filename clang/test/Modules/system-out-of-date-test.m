// RUN: rm -rf %t.cache
// RUN: echo '@import X;' | \
// RUN:   %clang_cc1 -fmodules -fimplicit-module-maps \
// RUN:     -fmodules-cache-path=%t.cache -I%S/Inputs/system-out-of-date \
// RUN:     -fsyntax-only -x objective-c -
//
// Build something with different diagnostic options.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t.cache -I%S/Inputs/system-out-of-date \
// RUN:   -fsyntax-only %s -Wnon-modular-include-in-framework-module \
// RUN:   -Werror=non-modular-include-in-framework-module 2>&1 \
// RUN: | FileCheck %s
@import X;

#import <Z.h>
// CHECK: While building module 'Z' imported from
// CHECK: {{.*}}Y-{{.*}}pcm' was validated as a system module and is now being imported as a non-system module
