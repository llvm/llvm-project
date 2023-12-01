// RUN: %clang_cl_asan  /EHsc /Od /std:c++17 %s -Fe%t
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %clang_cl_asan /EHsc /Od %p/dll_host.cpp -Fe%t
// RUN: %clang_cl_asan /EHsc /LD /Od /std:c++17 /DTEST_DLL %s -Fe%t.dll
// RUN: %run %t %t.dll 2>&1 | FileCheck %s

#include "operator_new_delete_replacement_macros.h"
#define DEFINED_REPLACEMENTS                                                   \
  (ALL_NEW | ALL_ALIGNED_NEW | SCALAR_DELETE | SCALAR_ALIGNED_DELETE)
#include "operator_new_delete_replacement_common.h"

// Covers:
// 9.  array (asan)                    -> scalar (custom)
// 10. nothrow (asan)                  -> scalar (custom)
// 11. sized (asan)                    -> scalar (custom) ** original bug report scenario **
// 13. sized array (asan)              -> array (asan)            -> scalar (custom)
// 15. array nothrow (asan)            -> array (asan)            -> scalar (custom)
// 16. aligned array (asan)            -> aligned scalar (custom)
// 17. aligned nothrow (asan)          -> aligned scalar (custom)
// 18. aligned sized (asan)            -> aligned scalar (custom)
// 20. aligned sized array (asan)      -> aligned array (asan)    -> aligned scalar (custom)
// 22. aligned array nothrow (asan)    -> aligned array (asan)    -> aligned scalar (custom)

// CHECK: new_scalar
// CHECK: new_array
// CHECK: new_scalar_nothrow
// CHECK: new_array_nothrow
// CHECK: delete_scalar
// CHECK: delete_scalar
// CHECK: delete_scalar
// CHECK: delete_scalar
// CHECK: new_scalar
// CHECK: new_array
// CHECK: delete_scalar
// CHECK: delete_scalar

// CHECK: new_scalar_align
// CHECK: new_array_align
// CHECK: new_scalar_align_nothrow
// CHECK: new_array_align_nothrow
// CHECK: delete_scalar_align
// CHECK: delete_scalar_align
// CHECK: delete_scalar_align
// CHECK: delete_scalar_align
// CHECK: new_scalar_align
// CHECK: new_array_align
// CHECK: delete_scalar_align
// CHECK: delete_scalar_align
