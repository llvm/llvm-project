// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'extern int a; template<typename T> int a2 = T::error;' > %t/a.h
// RUN: echo 'extern int b;' > %t/b.h
// RUN: echo 'extern int c = 0;' > %t/c.h
// RUN: echo 'module a { header "a.h" header "b.h" header "c.h" }' > %t/modulemap
// RUN: echo 'module other {}' > %t/other.modulemap

// We lazily check that the files referenced by an explicitly-specified .pcm
// file exist. Test this by removing files and ensuring that the compilation
// still succeeds.
//
// RUN: %clang_cc1 -fmodules -I %t -emit-module -fmodule-name=a -x c++ %t/modulemap -o %t/a.pcm \
// RUN:            -fmodule-map-file=%t/other.modulemap \
// RUN:            -fmodules-embed-file=%t/modulemap -fmodules-embed-file=%t/other.modulemap
// RUN: %clang_cc1 -fmodules -I %t -emit-module -fmodule-name=a -x c++ %t/modulemap -o %t/b.pcm \
// RUN:            -fmodule-map-file=%t/other.modulemap \
// RUN:            -fmodules-embed-all-files
// RUN: %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s
// RUN: not %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s -DERRORS 2>&1 | FileCheck %s
// RUN: rm %t/modulemap
// RUN: %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s
// RUN: not %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s -DERRORS 2>&1 | FileCheck %s
// RUN: rm %t/other.modulemap
// RUN: %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s
// RUN: not %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s -DERRORS 2>&1 | FileCheck %s
// RUN: sleep 1
// RUN: touch %t/a.h
// RUN: %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s
// RUN: not %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s -DERRORS 2>&1 | FileCheck %s
// RUN: rm %t/b.h
// RUN: %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s
// RUN: %clang_cc1 -fmodules -I %t -fmodule-file=%t/b.pcm %s
// RUN: not %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s -DERRORS 2>&1 | FileCheck %s --check-prefix=MISSING-B
// RUN: rm %t/a.h
// RUN: %clang_cc1 -fmodules -I %t -fmodule-file=%t/a.pcm %s -verify
// RUN: %clang_cc1 -fmodules -I %t -fmodule-file=%t/b.pcm %s -verify

// Oftentimes on Windows there are open handles, and deletion will fail.
// REQUIRES: can-remove-opened-file

#include "a.h" // expected-error {{file not found}}
int x = b;

#ifdef ERRORS
int y = a2<int>;
// CHECK: In module 'a':
// CHECK-NEXT: a.h:1:45: error:

// MISSING-B: could not find file '{{.*}}b.h'
// MISSING-B-NOT: please delete the module cache
#endif

// RUN: not %clang_cc1 -fmodules -I %t -emit-module -fmodule-name=a -x c++ /dev/null -o %t/c.pcm \
// RUN:                -fmodules-embed-file=%t/does-not-exist 2>&1 | FileCheck %s --check-prefix=MISSING-EMBED
// MISSING-EMBED: fatal error: file '{{.*}}does-not-exist' specified by '-fmodules-embed-file=' not found
