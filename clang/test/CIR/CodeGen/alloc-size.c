// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

#define NULL ((void *)0)

typedef unsigned long size_t;

// CIR: cir.func{{.*}}@my_malloc(!s32i){{.*}} attributes {allocsize = array<i32: 0>}
extern void *my_malloc(int) __attribute__((alloc_size(1)));
// CIR: cir.func{{.*}}@my_calloc(!s32i, !s32i){{.*}} attributes {allocsize = array<i32: 0, 1>}
extern void *my_calloc(int, int) __attribute__((alloc_size(1, 2)));

// CIR-LABEL: @call_direct
// LLVM-LABEL: @call_direct
void call_direct(void) {
  my_malloc(50);
  // CIR: cir.call @my_malloc(%{{.*}}) {allocsize = array<i32: 0>}
  // LLVM: call ptr @my_malloc(i32{{.*}} 50) [[DIRECT_MALLOC_ATTR:#[0-9]+]]
  my_calloc(1, 16);
  // CIR: cir.call @my_calloc(%{{.*}}) {allocsize = array<i32: 0, 1>}
  // LLVM: call ptr @my_calloc(i32{{.*}} 1, i32{{.*}} 16) [[DIRECT_CALLOC_ATTR:#[0-9]+]]
}

extern void *(*malloc_function_pointer)(void *, int)__attribute__((alloc_size(2)));
extern void *(*calloc_function_pointer)(void *, int, int)__attribute__((alloc_size(2, 3)));

// CIR-LABEL: @call_function_pointer
// LLVM-LABEL: @call_function_pointer
void call_function_pointer(void) {
  malloc_function_pointer(NULL, 100);
  // CIR: %[[MALLOC_FN_PTR_GLOBAL:.*]] = cir.get_global @malloc_function_pointer
  // CIR: %[[MALLOC_FN_PTR:.+]] = cir.load{{.*}}%[[MALLOC_FN_PTR_GLOBAL]]
  // CIR: cir.call %[[MALLOC_FN_PTR]]({{.*}}) {allocsize = array<i32: 1>}
  //
  // LLVM: %[[MALLOC_FN_PTR:.+]] = load ptr, ptr @malloc_function_pointer, align 8
  // LLVM: call ptr %[[MALLOC_FN_PTR]](ptr{{.*}} null, i32{{.*}} 100) [[INDIRECT_MALLOC_ATTR:#[0-9]+]]
  calloc_function_pointer(NULL, 2, 4);
  // CIR: %[[CALLOC_FN_PTR_GLOBAL:.*]] = cir.get_global @calloc_function_pointer
  // CIR: %[[CALLOC_FN_PTR:.+]] = cir.load{{.*}}%[[CALLOC_FN_PTR_GLOBAL]]
  // CIR: cir.call %[[CALLOC_FN_PTR]]({{.*}}) {allocsize = array<i32: 1, 2>}
  //
  // LLVM: %[[CALLOC_FN_PTR:.+]] = load ptr, ptr @calloc_function_pointer, align 8
  // LLVM: call ptr %[[CALLOC_FN_PTR]](ptr{{.*}} null, i32{{.*}} 2, i32{{.*}} 4) [[INDIRECT_CALLOC_ATTR:#[0-9]+]]
}

typedef void *(__attribute__((alloc_size(3))) * my_malloc_fn_pointer_type)(void *, void *, int);
typedef void *(__attribute__((alloc_size(3, 4))) * my_calloc_fn_pointer_type)(void *, void *, int, int);
extern my_malloc_fn_pointer_type malloc_function_pointer_with_typedef;
extern my_calloc_fn_pointer_type calloc_function_pointer_with_typedef;

// CIR-LABEL: @call_function_pointer_typedef
// LLVM-LABEL: @call_function_pointer_typedef
void call_function_pointer_typedef(void) {
  malloc_function_pointer_with_typedef(NULL, NULL, 200);
  // CIR: %[[INDIRECT_TYPEDEF_MALLOC_FN_PTR_GLOBAL:.*]] = cir.get_global @malloc_function_pointer_with_typedef
  // CIR: %[[INDIRECT_TYPEDEF_MALLOC_FN_PTR:.+]] = cir.load{{.*}}%[[INDIRECT_TYPEDEF_MALLOC_FN_PTR_GLOBAL]]
  // CIR: cir.call %[[INDIRECT_TYPEDEF_MALLOC_FN_PTR]]({{.*}}) {allocsize = array<i32: 2>}
  //
  // LLVM: %[[INDIRECT_TYPEDEF_MALLOC_FN_PTR:.+]] = load ptr, ptr @malloc_function_pointer_with_typedef, align 8
  // LLVM: call ptr %[[INDIRECT_TYPEDEF_MALLOC_FN_PTR]](ptr{{.*}} null, ptr{{.*}} null, i32{{.*}} 200) [[INDIRECT_TYPEDEF_MALLOC_ATTR:#[0-9]+]]
  calloc_function_pointer_with_typedef(NULL, NULL, 8, 4);
  // CIR: %[[INDIRECT_TYPEDEF_CALLOC_FN_PTR_GLOBAL:.*]] = cir.get_global @calloc_function_pointer_with_typedef
  // CIR: %[[INDIRECT_TYPEDEF_CALLOC_FN_PTR:.+]] = cir.load{{.*}}%[[INDIRECT_TYPEDEF_CALLOC_FN_PTR_GLOBAL]]
  // CIR: cir.call %[[INDIRECT_TYPEDEF_CALLOC_FN_PTR]]({{.*}}) {allocsize = array<i32: 2, 3>}
  //
  // LLVM: %[[INDIRECT_TYPEDEF_CALLOC_FN_PTR:.+]] = load ptr, ptr @calloc_function_pointer_with_typedef, align 8
  // LLVM: call ptr %[[INDIRECT_TYPEDEF_CALLOC_FN_PTR]](ptr{{.*}} null, ptr{{.*}} null, i32{{.*}} 8, i32{{.*}} 4) [[INDIRECT_TYPEDEF_CALLOC_ATTR:#[0-9]+]]
}

// LLVM: attributes [[DIRECT_MALLOC_ATTR]] = { allocsize(0) }
// LLVM: attributes [[DIRECT_CALLOC_ATTR]] = { allocsize(0,1) }
// LLVM: attributes [[INDIRECT_MALLOC_ATTR]] = { allocsize(1) }
// LLVM: attributes [[INDIRECT_CALLOC_ATTR]] = { allocsize(1,2) }
// LLVM: attributes [[INDIRECT_TYPEDEF_MALLOC_ATTR]] = { allocsize(2) }
// LLVM: attributes [[INDIRECT_TYPEDEF_CALLOC_ATTR]] = { allocsize(2,3) }
