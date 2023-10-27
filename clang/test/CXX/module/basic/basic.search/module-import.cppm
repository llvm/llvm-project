// Tests for imported module search.
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/x.cppm -o %t/x.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -fmodule-file=x=%t/x.pcm %t/y.cppm -o %t/y.pcm
//
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x=%t/x.pcm -verify %t/use.cpp \
// RUN:            -DMODULE_NAME=x
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=y=%t/y.pcm -verify %t/use.cpp \
// RUN:            -DMODULE_NAME=y -fmodule-file=x=%t/x.pcm
//
// RUN: mv %t/x.pcm %t/a.pcm
//
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=x=%t/a.pcm -verify %t/use.cpp \
// RUN:            -DMODULE_NAME=x
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=y=%t/y.pcm -fmodule-file=x=%t/a.pcm -verify %t/use.cpp \
// RUN:            -DMODULE_NAME=y
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -fmodule-file=y=%t/y.pcm -fmodule-file=x=%t/a.pcm %t/z.cppm -o %t/z.pcm
//
// RUN: %clang_cc1 -std=c++20 -I%t -fmodule-file=z=%t/z.pcm -fmodule-file=y=%t/y.pcm -fmodule-file=x=%t/a.pcm -verify %t/use.cpp \
// RUN:            -DMODULE_NAME=z
//

//--- x.cppm
export module x;
int a, b;

//--- y.cppm
export module y;
import x;
int c;

//--- z.cppm
export module z;
import y;
int d;

//--- use.cpp
import MODULE_NAME;

// expected-no-diagnostics
