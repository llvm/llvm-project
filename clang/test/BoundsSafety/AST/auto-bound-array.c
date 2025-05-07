
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

void foo(void) {
  // CHECK: VarDecl {{.+}} array_of_ptrs 'int *__single[2]'
  int *array_of_ptrs[2];

  // CHECK: VarDecl {{.+}} ptr_to_array 'int (*__bidi_indexable)[3]'
  int(*ptr_to_array)[3];

  // CHECK: VarDecl {{.+}} array_of_ptrs_to_arrays 'int (*__single[2])[3]'
  int(*array_of_ptrs_to_arrays[2])[3];

  // CHECK: VarDecl {{.+}} ptr_to_ptr_to_array 'int (*__single*__bidi_indexable)[3]'
  int(**ptr_to_ptr_to_array)[3];

  // CHECK: VarDecl {{.+}} ptr_to_array_of_ptrs 'int *__single(*__bidi_indexable)[3]'
  int *(*ptr_to_array_of_ptrs)[3];

  // CHECK: VarDecl {{.+}} ptr_to_array_of_ptrs_to_arrays 'int (*__single(*__bidi_indexable)[4])[3]'
  int(*(*ptr_to_array_of_ptrs_to_arrays)[4])[3];
}
