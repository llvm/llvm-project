// Tests for imported module search.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: echo 'export module x; int a, b;' > %t/x.cppm
// RUN: echo 'export module y; import x; int c;' > %t/y.cppm
// RUN: echo 'export module z; import y; int d;' > %t/z.cppm
//
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface %t/x.cppm -o %t/x.pcm
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface -fmodule-file=%t/x.pcm %t/y.cppm -o %t/y.pcm
//
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/x.pcm -verify %s \
// RUN:            -DMODULE_NAME=x
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/y.pcm -verify %s \
// RUN:            -DMODULE_NAME=y
//
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=x=%t/x.pcm -verify %s \
// RUN:            -DMODULE_NAME=x
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=y=%t/y.pcm -verify %s \
// RUN:            -DMODULE_NAME=y
//
// RUN: mv %t/x.pcm %t/a.pcm
//
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=x=%t/a.pcm -verify %s \
// RUN:            -DMODULE_NAME=x
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=%t/y.pcm -fmodule-file=x=%t/a.pcm -verify %s \
// RUN:            -DMODULE_NAME=y
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=y=%t/y.pcm -fmodule-file=x=%t/a.pcm -verify %s \
// RUN:            -DMODULE_NAME=y
//
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface -fmodule-file=y=%t/y.pcm -fmodule-file=x=%t/a.pcm %t/z.cppm -o %t/z.pcm
//
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -I%t -fmodule-file=z=%t/z.pcm -fmodule-file=y=%t/y.pcm -fmodule-file=x=%t/a.pcm -verify %s \
// RUN:            -DMODULE_NAME=z
//

import MODULE_NAME;

// expected-no-diagnostics
