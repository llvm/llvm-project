// RUN: rm -rf %t
// RUN: echo '@import X;' | \
// RUN:   %clang_cc1 -fmodules -fimplicit-module-maps \
// RUN:     -fmodules-cache-path=%t -I %S/Inputs/system-out-of-date \
// RUN:     -fsyntax-only -x objective-c -

// We have an version built with different diagnostic options.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs/system-out-of-date -Wnon-modular-include-in-framework-module -Werror=non-modular-include-in-framework-module %s -fsyntax-only 2>&1 | FileCheck %s
 @import X;
 
 #import <Z.h>
// CHECK: While building module 'Z' imported from
// CHECK: {{.*}}Y-{{.*}}pcm' was validated as a system module and is now being imported as a non-system module
