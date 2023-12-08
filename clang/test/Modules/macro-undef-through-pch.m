// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c-header -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:            -I%S/Inputs/macro-undef-through-pch -emit-pch \
// RUN:            %S/Inputs/macro-undef-through-pch/foo.h -o %t.pch
// RUN: %clang_cc1 -x objective-c -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:            -I%S/Inputs/macro-undef-through-pch -emit-pch \
// RUN:            -include-pch %t.pch %s

// PR19215
#undef AB
