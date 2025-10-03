// This test checks that we don't crash when we load two conflicting PCM files
// and instead emit the appropriate diagnostics.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: mkdir %t/frameworks1

// RUN: clang-scan-deps -format experimental-full -o %t/deps1.json -- \
// RUN:   %clang -fmodules -fmodules-cache-path=%t/cache \
// RUN:   -F %t/frameworks1 -F %t/frameworks2 \
// RUN:   -c %t/tu1.m -o %t/tu1.o

// RUN: cp -r %t/frameworks2/A.framework %t/frameworks1

// RUN: not clang-scan-deps -format experimental-full -o %t/deps2.json 2>%t/errs -- \
// RUN:   %clang -fmodules -fmodules-cache-path=%t/cache \
// RUN:   -F %t/frameworks1 -F %t/frameworks2 \
// RUN:   -c %t/tu2.m -o %t/tu2.o
// RUN: FileCheck --input-file=%t/errs %s

// CHECK:      fatal error: module 'A' is defined in both '{{.*}}.pcm' and '{{.*}}.pcm'
// CHECK-NEXT: note: compiled from '{{.*}}frameworks1{{.*}}' and '{{.*}}frameworks2{{.*}}'

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
#include <B/B.h> // This results in a conflict and a fatal loader error.

#if MACRO_A // This crashes with lexer that does not respect `cutOfLexing()`.
#endif
