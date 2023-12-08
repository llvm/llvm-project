// RUN: %clang_cc1 -E %s -I%S/Inputs | FileCheck -strict-whitespace %s
// RUN: %clang_cc1 -fms-compatibility -DMS -E %s -I%S/Inputs | FileCheck -check-prefix=CHECK-MS -strict-whitespace %s 
// RUN: %clang_cc1 -E %s -I%S/Inputs -DBADINC -verify

#ifdef BADINC

// Paranoia.

__FILE_NAME__
#include <include-subdir/> // expected-error {{file not found}}
__FILE_NAME__

#else

// Reference.
1: "file_name_macro.c"

// Ensure it expands correctly for this file.
2: __FILE_NAME__

// CHECK: {{^}}1: "file_name_macro.c"
// CHECK: {{^}}2: "file_name_macro.c"

// Test if inclusion works right.
#ifdef MS
#include <include-subdir\file_name_macro_include.h>
// MS compatibility allows for mixed separators in paths.
#include <include-subdir/subdir1\hdr1.h>
#include <include-subdir\subdir1/hdr2.h>
#else
#include <include-subdir/file_name_macro_include.h>
#endif

#include <include-subdir/h>

// CHECK: {{^}}3: "file_name_macro_include.h"
// CHECK: {{^}}4: "file_name_macro_include.h"
// CHECK-NOT: {{^}}5: "file_name_macro_include.h"
// CHECK-MS: {{^}}5: "file_name_macro_include.h"
// CHECK: {{^}}6: "h"
// CHECK-MS: {{^}}7: "hdr1.h"
// CHECK-MS: {{^}}8: "hdr2.h"

#endif
