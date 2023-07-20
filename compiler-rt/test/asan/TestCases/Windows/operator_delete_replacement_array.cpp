// RUN: %clang_cl_asan /EHsc /Od /std:c++17 %s -Fe%t
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %clang_cl_asan /EHsc /Od %p/dll_host.cpp -Fe%t
// RUN: %clang_cl_asan /EHsc /LD /Od /std:c++17 /DTEST_DLL %s -Fe%t.dll
// RUN: %run %t %t.dll 2>&1 | FileCheck %s

#include "operator_new_delete_replacement_macros.h"
#define DEFINED_REPLACEMENTS                                                   \
  (ALL_NEW | ALL_ALIGNED_NEW | SCALAR_DELETE | ARRAY_DELETE |                  \
   SCALAR_ALIGNED_DELETE | ARRAY_ALIGNED_DELETE)
#include "operator_new_delete_replacement_common.h"

// Covers:
// 12. sized array (asan)              -> array (custom)
// 14. array nothrow (asan)            -> array (custom)
// 19. aligned sized array (asan)      -> aligned array (custom)
// 21. aligned array nothrow (asan)    -> aligned array (custom)

// CHECK: new_scalar
// CHECK: new_array
// CHECK: new_scalar_nothrow
// CHECK: new_array_nothrow
// CHECK: delete_scalar
// CHECK: delete_array
// CHECK: delete_scalar
// CHECK: delete_array
// CHECK: new_scalar
// CHECK: new_array
// CHECK: delete_scalar
// CHECK: delete_array

// CHECK: new_scalar_align
// CHECK: new_array_align
// CHECK: new_scalar_align_nothrow
// CHECK: new_array_align_nothrow
// CHECK: delete_scalar_align
// CHECK: delete_array_align
// CHECK: delete_scalar_align
// CHECK: delete_array_align
// CHECK: new_scalar_align
// CHECK: new_array_align
// CHECK: delete_scalar_align
// CHECK: delete_array_align
