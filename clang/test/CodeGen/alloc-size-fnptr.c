// Check that the alloc_size attribute is propagated to the call instruction
// for both direct and indirect calls
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin -emit-llvm %s -o - 2>&1 | FileCheck %s

#define NULL ((void *)0)

int gi;

typedef unsigned long size_t;

extern void *my_malloc(int) __attribute__((alloc_size(1)));
extern void *my_calloc(int, int) __attribute__((alloc_size(1, 2)));

// CHECK-LABEL: @call_direct
void call_direct(void) {
  my_malloc(50);
  // CHECK: call i8* @my_malloc(i32 noundef 50) [[DIRECT_MALLOC_ATTR:#[0-9]+]]
  my_calloc(1, 16);
  // CHECK: call i8* @my_calloc(i32 noundef 1, i32 noundef 16) [[DIRECT_CALLOC_ATTR:#[0-9]+]]
}

extern void *(*malloc_function_pointer)(void *, int)__attribute__((alloc_size(2)));
extern void *(*calloc_function_pointer)(void *, int, int)__attribute__((alloc_size(2, 3)));

// CHECK-LABEL: @call_function_pointer
void call_function_pointer(void) {
  malloc_function_pointer(NULL, 100);
  // CHECK: [[MALLOC_FN_PTR:%.+]] = load i8* (i8*, i32)*, i8* (i8*, i32)** @malloc_function_pointer, align 8
  // CHECK: call i8* [[MALLOC_FN_PTR]](i8* noundef null, i32 noundef 100) [[INDIRECT_MALLOC_ATTR:#[0-9]+]]
  calloc_function_pointer(NULL, 2, 4);
  // CHECK: [[CALLOC_FN_PTR:%.+]] = load i8* (i8*, i32, i32)*, i8* (i8*, i32, i32)** @calloc_function_pointer, align 8
  // CHECK: call i8* [[CALLOC_FN_PTR]](i8* noundef null, i32 noundef 2, i32 noundef 4) [[INDIRECT_CALLOC_ATTR:#[0-9]+]]
}

typedef void *(__attribute__((alloc_size(3))) * my_malloc_fn_pointer_type)(void *, void *, int);
typedef void *(__attribute__((alloc_size(3, 4))) * my_calloc_fn_pointer_type)(void *, void *, int, int);
extern my_malloc_fn_pointer_type malloc_function_pointer_with_typedef;
extern my_calloc_fn_pointer_type calloc_function_pointer_with_typedef;

// CHECK-LABEL: @call_function_pointer_typedef
void call_function_pointer_typedef(void) {
  malloc_function_pointer_with_typedef(NULL, NULL, 200);
  // CHECK: [[INDIRECT_TYPEDEF_MALLOC_FN_PTR:%.+]] = load i8* (i8*, i8*, i32)*, i8* (i8*, i8*, i32)** @malloc_function_pointer_with_typedef, align 8
  // CHECK: call i8* [[INDIRECT_TYPEDEF_MALLOC_FN_PTR]](i8* noundef null, i8* noundef null, i32 noundef 200) [[INDIRECT_TYPEDEF_MALLOC_ATTR:#[0-9]+]]
  calloc_function_pointer_with_typedef(NULL, NULL, 8, 4);
  // CHECK: [[INDIRECT_TYPEDEF_CALLOC_FN_PTR:%.+]] = load i8* (i8*, i8*, i32, i32)*, i8* (i8*, i8*, i32, i32)** @calloc_function_pointer_with_typedef, align 8
  // CHECK: call i8* [[INDIRECT_TYPEDEF_CALLOC_FN_PTR]](i8* noundef null, i8* noundef null, i32 noundef 8, i32 noundef 4) [[INDIRECT_TYPEDEF_CALLOC_ATTR:#[0-9]+]]
}

// CHECK: attributes [[DIRECT_MALLOC_ATTR]] = { allocsize(0) }
// CHECK: attributes [[DIRECT_CALLOC_ATTR]] = { allocsize(0,1) }
// CHECK: attributes [[INDIRECT_MALLOC_ATTR]] = { allocsize(1) }
// CHECK: attributes [[INDIRECT_CALLOC_ATTR]] = { allocsize(1,2) }
// CHECK: attributes [[INDIRECT_TYPEDEF_MALLOC_ATTR]] = { allocsize(2) }
// CHECK: attributes [[INDIRECT_TYPEDEF_CALLOC_ATTR]] = { allocsize(2,3) }
