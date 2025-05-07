

// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// Functions

// CHECK: FunctionDecl {{.+}} func_proto_void_cdecl 'void (void) __attribute__((cdecl))':'void (void)'
__attribute__((cdecl)) void func_proto_void_cdecl(void);

// CHECK: FunctionDecl {{.+}} func_proto_void_noreturn 'void (void) __attribute__((noreturn))'
__attribute__((noreturn)) void func_proto_void_noreturn(void);

// CHECK: FunctionDecl {{.+}} func_noproto_void_cdecl 'void () __attribute__((cdecl))':'void ()'
__attribute__((cdecl)) void func_noproto_void_cdecl();

// CHECK: FunctionDecl {{.+}} func_noproto_void_noreturn 'void () __attribute__((noreturn))'
__attribute__((noreturn)) void func_noproto_void_noreturn();

// CHECK: FunctionDecl {{.+}} func_proto_ret_cdecl 'void *__single(void) __attribute__((cdecl))':'void *__single(void)'
__attribute__((cdecl)) void *func_proto_ret_cdecl(void);

// CHECK: FunctionDecl {{.+}} func_proto_ret_noreturn 'void *__single(void) __attribute__((noreturn))'
__attribute__((noreturn)) void *func_proto_ret_noreturn(void);

// CHECK: FunctionDecl {{.+}} func_noproto_ret_cdecl 'void *__single() __attribute__((cdecl))':'void *__single()'
__attribute__((cdecl)) void *func_noproto_ret_cdecl();

// CHECK: FunctionDecl {{.+}} func_noproto_ret_noreturn 'void *__single() __attribute__((noreturn))'
__attribute__((noreturn)) void *func_noproto_ret_noreturn();

// Function pointers

// CHECK: VarDecl {{.+}} fptr_proto_void_cdecl 'void ((*__single))(void) __attribute__((cdecl))'
__attribute__((cdecl)) void (*fptr_proto_void_cdecl)(void);

// CHECK: VarDecl {{.+}} fptr_proto_void_noreturn 'void (*__single)(void) __attribute__((noreturn))'
__attribute__((noreturn)) void (*fptr_proto_void_noreturn)(void);

// CHECK: VarDecl {{.+}} fptr_noproto_void_cdecl 'void ((*__single))() __attribute__((cdecl))'
__attribute__((cdecl)) void (*fptr_noproto_void_cdecl)();

// CHECK: VarDecl {{.+}} fptr_noproto_void_noreturn 'void (*__single)() __attribute__((noreturn))'
__attribute__((noreturn)) void (*fptr_noproto_void_noreturn)();

// CHECK: VarDecl {{.+}} fptr_proto_ret_cdecl 'void *__single((*__single))(void) __attribute__((cdecl))'
__attribute__((cdecl)) void *(*fptr_proto_ret_cdecl)(void);

// CHECK: VarDecl {{.+}} fptr_proto_ret_noreturn 'void *__single(*__single)(void) __attribute__((noreturn))'
__attribute__((noreturn)) void *(*fptr_proto_ret_noreturn)(void);

// CHECK: VarDecl {{.+}} fptr_noproto_ret_cdecl 'void *__single((*__single))() __attribute__((cdecl))'
__attribute__((cdecl)) void *(*fptr_noproto_ret_cdecl)();

// CHECK: VarDecl {{.+}} fptr_noproto_ret_noreturn 'void *__single(*__single)() __attribute__((noreturn))'
__attribute__((noreturn)) void *(*fptr_noproto_ret_noreturn)();
