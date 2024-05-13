// RUN: rm -rf %t
// RUN: mkdir %t
//
// RUN: %clang_cc1 -x objective-c-header -emit-pch %S/Inputs/pch-typedef.h -o %t/pch-typedef.h.gch \
// RUN:   -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/one
//
// RUN: not %clang_cc1 -x objective-c -fsyntax-only %s -include-pch %t/pch-typedef.h.gch \
// RUN:   -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/two 2>&1 \
// RUN:   | FileCheck %s -check-prefixes=CHECK-ERROR
//
// RUN: %clang_cc1 -x objective-c -fsyntax-only %s -include-pch %t/pch-typedef.h.gch \
// RUN:   -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/two 2>&1 -fallow-pch-with-different-modules-cache-path \
// RUN:   | FileCheck %s -check-prefixes=CHECK-SUCCESS -allow-empty

pch_int x = 0;

// CHECK-ERROR: PCH was compiled with module cache path '{{.*}}', but the path is currently '{{.*}}'
// CHECK-SUCCESS-NOT: PCH was compiled with module cache path '{{.*}}', but the path is currently '{{.*}}'
