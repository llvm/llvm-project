// RUN: %clang_cl_asan /EHsc /Od /std:c++17 %s -Fe%t
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %clang_cl_asan /EHsc /Od %p/dll_host.cpp -Fe%t
// RUN: %clang_cl_asan /EHsc /LD /Od /std:c++17 /DTEST_DLL %s -Fe%t.dll
// RUN: %run %t %t.dll 2>&1 | FileCheck %s

#include "operator_new_delete_replacement_macros.h"
#define DEFINED_REPLACEMENTS                                                   \
  (SCALAR_NEW | ARRAY_NEW | SCALAR_ALIGNED_NEW | ARRAY_ALIGNED_NEW |           \
   ALL_DELETE | ALL_ALIGNED_DELETE)
#include "operator_new_delete_replacement_common.h"

// Covers:
// 3. array nothrow (asan)            -> array  (custom)
// 7. aligned array nothrow (asan)    -> aligned array  (custom)

// CHECK: new_scalar
// CHECK: new_array
// CHECK: new_scalar
// CHECK: new_array
// CHECK: delete_scalar
// CHECK: delete_array
// CHECK: delete_scalar_nothrow
// CHECK: delete_array_nothrow
// CHECK: new_scalar
// CHECK: new_array
// CHECK: delete_scalar_size
// CHECK: delete_array_size

// CHECK: new_scalar_align
// CHECK: new_array_align
// CHECK: new_scalar_align
// CHECK: new_array_align
// CHECK: delete_scalar_align
// CHECK: delete_array_align
// CHECK: delete_scalar_align_nothrow
// CHECK: delete_array_align_nothrow
// CHECK: new_scalar_align
// CHECK: new_array_align
// CHECK: delete_scalar_size_align
// CHECK: delete_array_size_align
