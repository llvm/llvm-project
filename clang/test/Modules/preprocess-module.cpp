// RUN: rm -rf %t

// RUN: not %clang_cc1 -fmodules -fmodule-name=file -I%S/Inputs/preprocess -x c++-module-map %S/Inputs/preprocess/module.modulemap -E 2>&1 | FileCheck %s --check-prefix=MISSING-FWD
// MISSING-FWD: module 'fwd' is needed

// RUN: %clang_cc1 -fmodules -fmodule-name=file -fmodules-cache-path=%t -I%S/Inputs/preprocess -x c++-module-map %S/Inputs/preprocess/module.modulemap -E | FileCheck %s --check-prefix=CHECK --check-prefix=NO-REWRITE
// RUN: %clang_cc1 -fmodules -fmodule-name=file -fmodules-cache-path=%t -I%S/Inputs/preprocess -x c++-module-map %S/Inputs/preprocess/module.modulemap -E -frewrite-includes | FileCheck %s --check-prefix=CHECK --check-prefix=REWRITE

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
