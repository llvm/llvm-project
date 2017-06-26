// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: not %clang_cc1 -fmodules -fmodule-name=file -I%S/Inputs/preprocess -x c++-module-map %S/Inputs/preprocess/module.modulemap -E 2>&1 | FileCheck %s --check-prefix=MISSING-FWD
// MISSING-FWD: module 'fwd' is needed

// RUN: %clang_cc1 -fmodules -fmodule-name=fwd -I%S/Inputs/preprocess -x c++-module-map %S/Inputs/preprocess/module.modulemap -emit-module -o %t/fwd.pcm

// Check that we can preprocess modules, and get the expected output.
// RUN: %clang_cc1 -fmodules -fmodule-name=file -fmodule-file=%t/fwd.pcm -I%S/Inputs/preprocess -x c++-module-map %S/Inputs/preprocess/module.modulemap -E -o %t/no-rewrite.ii
// RUN: %clang_cc1 -fmodules -fmodule-name=file -fmodule-file=%t/fwd.pcm -I%S/Inputs/preprocess -x c++-module-map %S/Inputs/preprocess/module.modulemap -E -frewrite-includes -o %t/rewrite.ii
//
// RUN: FileCheck %s --input-file %t/no-rewrite.ii --check-prefix=CHECK --check-prefix=NO-REWRITE
// RUN: FileCheck %s --input-file %t/rewrite.ii    --check-prefix=CHECK --check-prefix=REWRITE

// Check that we can build a module from the preprocessed output.
// RUN: %clang_cc1 -fmodules -fmodule-name=file -fmodule-file=%t/fwd.pcm -x c++-module-map-cpp-output %t/no-rewrite.ii -emit-module -o %t/no-rewrite.pcm
// RUN: %clang_cc1 -fmodules -fmodule-name=file -fmodule-file=%t/fwd.pcm -x c++-module-map-cpp-output %t/rewrite.ii -emit-module -o %t/rewrite.pcm

// Check that we can load the original module map in the same compilation (this
// could happen if we had a redundant -fmodule-map-file= in the original
// build).
// RUN: %clang_cc1 -fmodules -fmodule-name=file -fmodule-file=%t/fwd.pcm -fmodule-map-file=%S/Inputs/preprocess/module.modulemap -x c++-module-map-cpp-output %t/rewrite.ii -emit-module -o /dev/null

// Check the module we built works.
// RUN: %clang_cc1 -fmodules -fmodule-file=%t/no-rewrite.pcm %s -I%t -verify -fno-modules-error-recovery
// RUN: %clang_cc1 -fmodules -fmodule-file=%t/rewrite.pcm %s -I%t -verify -fno-modules-error-recovery -DREWRITE
// RUN: %clang_cc1 -fmodules -fmodule-file=%t/no-rewrite.pcm %s -I%t -verify -fno-modules-error-recovery -DINCLUDE -I%S/Inputs/preprocess
// RUN: %clang_cc1 -fmodules -fmodule-file=%t/rewrite.pcm %s -I%t -verify -fno-modules-error-recovery -DREWRITE -DINCLUDE -I%S/Inputs/preprocess

// Now try building the module when the header files are missing.
// RUN: cp %S/Inputs/preprocess/fwd.h %S/Inputs/preprocess/file.h %S/Inputs/preprocess/file2.h %S/Inputs/preprocess/other.h %S/Inputs/preprocess/module.modulemap %t
// RUN: %clang_cc1 -fmodules -fmodule-name=file -fmodule-file=%t/fwd.pcm -I%t -x c++-module-map %t/module.modulemap -E -frewrite-includes -o %t/copy.ii
// RUN: rm %t/fwd.h %t/file.h %t/file2.h %t/other.h %t/module.modulemap
// RUN: %clang_cc1 -fmodules -fmodule-name=file -fmodule-file=%t/fwd.pcm -x c++-module-map-cpp-output %t/copy.ii -emit-module -o %t/copy.pcm

// Check that our module contains correct mapping information for the headers.
// RUN: cp %S/Inputs/preprocess/fwd.h %S/Inputs/preprocess/file.h %S/Inputs/preprocess/file2.h %S/Inputs/preprocess/other.h %S/Inputs/preprocess/module.modulemap %t
// RUN: %clang_cc1 -fmodules -fmodule-file=%t/copy.pcm %s -I%t -verify -fno-modules-error-recovery -DCOPY -DINCLUDE
// RUN: rm %t/fwd.h %t/file.h %t/file2.h %t/other.h %t/module.modulemap

// Check that we can preprocess from a .pcm file and that we get the same result as preprocessing from the original sources.
// RUN: %clang_cc1 -fmodules -fmodule-name=file -fmodule-file=%t/fwd.pcm -I%S/Inputs/preprocess -x c++-module-map %S/Inputs/preprocess/module.modulemap -emit-module -o %t/file.pcm
// RUN: %clang_cc1 -fmodules -fmodule-name=file -fmodule-file=%t/fwd.pcm -I%S/Inputs/preprocess %t/file.pcm -E -frewrite-includes -o %t/file.rewrite.ii
// FIXME: This check fails on Windows targets, due to canonicalization of directory separators.
// FIXME: cmp %t/rewrite.ii %t/file.rewrite.ii
// FIXME: Instead, just check that the preprocessed output is functionally equivalent to the output when preprocessing from the original sources.
// RUN: FileCheck %s --input-file %t/file.rewrite.ii    --check-prefix=CHECK --check-prefix=REWRITE
// RUN: %clang_cc1 -fmodules -fmodule-name=file -fmodule-file=%t/fwd.pcm -x c++-module-map-cpp-output %t/file.rewrite.ii -emit-module -o %t/file.rewrite.pcm
// RUN: %clang_cc1 -fmodules -fmodule-file=%t/file.rewrite.pcm %s -I%t -verify -fno-modules-error-recovery -DFILE_REWRITE
// RUN: %clang_cc1 -fmodules -fmodule-file=%t/file.rewrite.pcm %s -I%t -verify -fno-modules-error-recovery -DFILE_REWRITE -DINCLUDE -I%S/Inputs/preprocess
//
// Check that we can preprocess this user of the .pcm file.
// RUN: %clang_cc1 -fmodules -fmodule-file=%t/file.pcm %s -I%t -E -frewrite-imports -o %t/preprocess-module.ii
// RUN: %clang_cc1 -fmodules %t/preprocess-module.ii -verify -fno-modules-error-recovery -DFILE_REWRITE_FULL
//
// Check that language / header search options are ignored when preprocessing from a .pcm file.
// RUN: %clang_cc1 %t/file.pcm -E -frewrite-includes -o %t/file.rewrite.ii.2
// RUN: cmp %t/file.rewrite.ii %t/file.rewrite.ii.2
//
// RUN: %clang_cc1 -fmodules -fmodule-name=file -fmodule-file=%t/fwd.pcm -I%S/Inputs/preprocess %t/file.pcm -E -o %t/file.no-rewrite.ii
// RUN: %clang_cc1 %t/file.pcm -E -o %t/file.no-rewrite.ii.2 -Dstruct=error
// RUN: cmp %t/file.no-rewrite.ii %t/file.no-rewrite.ii.2

// == module map
// CHECK: # 1 "{{.*}}module.modulemap"
// CHECK: module file {
// CHECK:   header "file.h" { size
// CHECK:   header "file2.h" { size
// CHECK: }

// == file.h
// CHECK: # 1 "<module-includes>"
// REWRITE: #if 0
// REWRITE: #include "file.h"
// REWRITE: #endif
//
// FIXME: It would be preferable to consistently put the module begin/end in
// the same file, but the relative ordering of PP callbacks and module
// begin/end tokens makes that difficult.
//
// REWRITE: #pragma clang module begin file
// CHECK: # 1 "{{.*}}file.h" 1
// NO-REWRITE: #pragma clang module begin file
//
// REWRITE: #ifndef FILE_H
// REWRITE: #define FILE_H
//
// CHECK: #pragma clang module import fwd /* clang {{-E|-frewrite-includes}}: implicit import
// CHECK: typedef struct __FILE FILE;
//
// REWRITE: #endif
//
// REWRITE: #pragma clang module end
// CHECK: # 2 "<module-includes>" 2
// NO-REWRITE: #pragma clang module end

// == file2.h
// REWRITE: #if 0
// REWRITE: #include "file2.h"
// REWRITE: #endif
//
// REWRITE: #pragma clang module begin file
// CHECK: # 1 "{{.*}}file2.h" 1
// NO-REWRITE: #pragma clang module begin file
//
// ==== recursively re-enter file.h
// REWRITE: #if 0
// REWRITE: #include "file.h"
// REWRITE: #endif
//
// REWRITE: #pragma clang module begin file
// CHECK: # 1 "{{.*}}file.h" 1
// NO-REWRITE: #pragma clang module begin file
//
// REWRITE: #ifndef FILE_H
// REWRITE: #define FILE_H
// REWRITE: #if 0
// REWRITE: #include "fwd.h"
// REWRITE: #endif
// REWRITE-NOT: #pragma clang module import fwd
// REWRITE: #endif
//
// NO-REWRITE-NOT: struct __FILE;
//
// REWRITE: #pragma clang module end
// CHECK: # 2 "{{.*}}file2.h" 2
// NO-REWRITE: #pragma clang module end
// NO-REWRITE: # 2 "{{.*}}file2.h"{{$}}
// ==== return to file2.h
//
// CHECK: extern int file2;
//
// REWRITE: #pragma clang module end
// CHECK: # 3 "<module-includes>" 2
// NO-REWRITE: #pragma clang module end


__FILE *a; // expected-error-re {{{{declaration of '__FILE' must be imported|unknown type name '__FILE'}}}}
#if FILE_REWRITE
// expected-note@file.rewrite.ii:* {{here}}
#elif FILE_REWRITE_FULL
// No note diagnostic at all in this case: we've built the 'file' module but not loaded it into this compilation yet.
#elif REWRITE
// expected-note@rewrite.ii:* {{here}}
#elif COPY
// expected-note@copy.ii:* {{here}}
#else
// expected-note@no-rewrite.ii:* {{here}}
#endif

#ifdef INCLUDE
#include "file.h"
#else
#pragma clang module import file
#endif

FILE *b;
int x = file2; // ok, found in file2.h, even under -DINCLUDE
