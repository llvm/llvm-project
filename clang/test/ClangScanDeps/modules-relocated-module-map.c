// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: mkdir %t/frameworks1

// RUN: clang-scan-deps -format experimental-full -- \
// RUN:   %clang -fmodules -fmodules-cache-path=%t/cache \
// RUN:   -F %t/frameworks1 -F %t/frameworks2 \
// RUN:   -c %t/tu1.m -o %t/tu1.o

// RUN: cp -r %t/frameworks2/A.framework %t/frameworks1

// RUN: clang-scan-deps -format experimental-full -- \
// RUN:   %clang -fmodules -fmodules-cache-path=%t/cache \
// RUN:   -F %t/frameworks1 -F %t/frameworks2 \
// RUN:   -c %t/tu2.m -o %t/tu2.o

//--- frameworks2/A.framework/Modules/module.modulemap
framework module A { header "A.h" }
//--- frameworks2/A.framework/Headers/A.h
#define MACRO_A 1

//--- frameworks2/B.framework/Modules/module.modulemap
framework module B { header "B.h" }
//--- frameworks2/B.framework/Headers/B.h
#include <A/A.h>

//--- tu1.m
#include <B/B.h>

//--- tu2.m
#include <A/A.h>
#include <B/B.h>

#if MACRO_A == 3
#endif
