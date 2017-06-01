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


// == module map
// CHECK: # 1 "{{.*}}module.modulemap"
// CHECK: module file {
// CHECK:   header "file.h"
// CHECK:   header "file2.h"
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
// NO-REWRITE: # 1 "{{.*}}file.h"{{$}}
//
// CHECK: struct __FILE;
// CHECK: #pragma clang module import fwd /* clang {{-E|-frewrite-includes}}: implicit import
// CHECK: typedef struct __FILE FILE;
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
// NO-REWRITE: # 1 "{{.*}}file.h"{{$}}
//
// CHECK: struct __FILE;
// CHECK: #pragma clang module import fwd /* clang {{-E|-frewrite-includes}}: implicit import
// CHECK: typedef struct __FILE FILE;
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


__FILE *a; // expected-error {{declaration of '__FILE' must be imported}}
#ifdef REWRITE
// expected-note@rewrite.ii:1 {{here}}
#else
// expected-note@no-rewrite.ii:1 {{here}}
#endif

#ifdef INCLUDE
#include "file.h"
#else
#pragma clang module import file
#endif

FILE *b;
int x = file2; // ok, found in file2.h, even under -DINCLUDE
